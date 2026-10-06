// Package db ties the index, the payload store and the write-ahead log into
// durable named collections.
package db

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"syscall"

	"github.com/rishiraj38/quiver/internal/vec"
	"github.com/rishiraj38/quiver/internal/wal"
)

var (
	ErrNotFound = errors.New("not found")
	ErrExists   = errors.New("already exists")
	ErrInvalid  = errors.New("invalid argument")
	ErrClosed   = errors.New("closed")
	// ErrLocked means another process has the data directory open.
	ErrLocked = errors.New("data directory is in use by another process")
)

const (
	metaName = "meta.json"
	lockName = "LOCK"
	maxDim   = 65536
)

var nameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`)

// Options configures a DB.
type Options struct {
	WAL wal.Options
}

// DB is a directory of collections, one sub-directory each.
type DB struct {
	dir  string
	opts Options
	lock *os.File

	mu          sync.RWMutex
	collections map[string]*Collection
}

// Open loads every collection found under dir, replaying their logs.
func Open(dir string, opts Options) (*DB, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	// Two servers appending to the same logs would corrupt them. The flock
	// is tied to the open file, so the kernel drops it if the process dies.
	lock, err := os.OpenFile(filepath.Join(dir, lockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		return nil, fmt.Errorf("%w: %s", ErrLocked, dir)
	}
	d := &DB{dir: dir, opts: opts, lock: lock, collections: make(map[string]*Collection)}
	items, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, it := range items {
		if !it.IsDir() || !nameRE.MatchString(it.Name()) {
			continue
		}
		cdir := filepath.Join(dir, it.Name())
		raw, err := os.ReadFile(filepath.Join(cdir, metaName))
		if errors.Is(err, os.ErrNotExist) {
			// A create that crashed before its meta file landed.
			continue
		}
		if err != nil {
			d.Close()
			return nil, err
		}
		var cfg CollectionConfig
		if err := json.Unmarshal(raw, &cfg); err != nil {
			d.Close()
			return nil, fmt.Errorf("collection %s: bad %s: %w", it.Name(), metaName, err)
		}
		c, err := openCollection(cdir, it.Name(), cfg, opts.WAL)
		if err != nil {
			d.Close()
			return nil, err
		}
		d.collections[it.Name()] = c
	}
	return d, nil
}

// Create makes a new empty collection.
func (d *DB) Create(name string, cfg CollectionConfig) (*Collection, error) {
	if !nameRE.MatchString(name) {
		return nil, fmt.Errorf("%w: collection name must match %s", ErrInvalid, nameRE)
	}
	if cfg.Dim <= 0 || cfg.Dim > maxDim {
		return nil, fmt.Errorf("%w: dim must be between 1 and %d", ErrInvalid, maxDim)
	}
	metric, err := vec.ParseMetric(cfg.Metric)
	if err != nil {
		return nil, fmt.Errorf("%w: metric must be l2, cosine or dot", ErrInvalid)
	}
	cfg.Metric = metric.String()
	if cfg.M == 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction == 0 {
		cfg.EfConstruction = 200
	}
	if cfg.M < 2 || cfg.M > 128 || cfg.EfConstruction < cfg.M || cfg.EfConstruction > 4096 {
		return nil, fmt.Errorf("%w: need 2 <= m <= 128 and m <= ef_construction <= 4096", ErrInvalid)
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.collections == nil {
		return nil, ErrClosed
	}
	if _, ok := d.collections[name]; ok {
		return nil, fmt.Errorf("%w: collection %s", ErrExists, name)
	}
	cdir := filepath.Join(d.dir, name)
	// Leftovers from a create or drop that did not finish.
	if err := os.RemoveAll(cdir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(cdir, 0o755); err != nil {
		return nil, err
	}
	raw, _ := json.MarshalIndent(cfg, "", "  ")
	if err := writeFileSync(filepath.Join(cdir, metaName), raw); err != nil {
		return nil, err
	}
	if err := syncDir(d.dir); err != nil {
		return nil, err
	}
	c, err := openCollection(cdir, name, cfg, d.opts.WAL)
	if err != nil {
		return nil, err
	}
	d.collections[name] = c
	return c, nil
}

// Get returns an open collection by name.
func (d *DB) Get(name string) (*Collection, error) {
	d.mu.RLock()
	c, ok := d.collections[name]
	d.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: collection %s", ErrNotFound, name)
	}
	return c, nil
}

// List returns the stats of every collection, ordered by name.
func (d *DB) List() []Stats {
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := make([]Stats, 0, len(d.collections))
	for _, name := range sortedIDs(d.collections) {
		out = append(out, d.collections[name].Stats())
	}
	return out
}

// Drop closes a collection and deletes its files.
func (d *DB) Drop(name string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	c, ok := d.collections[name]
	if !ok {
		return fmt.Errorf("%w: collection %s", ErrNotFound, name)
	}
	delete(d.collections, name)
	cdir := filepath.Join(d.dir, name)
	// Removing the meta file first makes the drop atomic: without it the
	// directory is ignored on the next Open even if RemoveAll is cut short.
	cerr := c.Close()
	if err := os.Remove(filepath.Join(cdir, metaName)); err != nil {
		return err
	}
	if err := os.RemoveAll(cdir); err != nil {
		return err
	}
	return cerr
}

// SnapshotAll snapshots every collection and returns the first error.
func (d *DB) SnapshotAll() error {
	d.mu.RLock()
	cs := make([]*Collection, 0, len(d.collections))
	for _, c := range d.collections {
		cs = append(cs, c)
	}
	d.mu.RUnlock()
	var first error
	for _, c := range cs {
		if err := c.Snapshot(); err != nil && !errors.Is(err, ErrClosed) && first == nil {
			first = fmt.Errorf("collection %s: %w", c.name, err)
		}
	}
	return first
}

// Close closes every collection. The DB cannot be used afterwards.
func (d *DB) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	var first error
	for _, c := range d.collections {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	d.collections = nil
	if d.lock != nil {
		d.lock.Close()
		d.lock = nil
	}
	return first
}

func writeFileSync(path string, data []byte) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
