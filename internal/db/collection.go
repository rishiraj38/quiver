package db

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/rishiraj38/quiver/internal/hnsw"
	"github.com/rishiraj38/quiver/internal/vec"
	"github.com/rishiraj38/quiver/internal/wal"
)

const (
	numStripes = 256
	// Batches at least this large are applied to the index by several
	// goroutines; below it the fan-out costs more than it saves.
	parallelApplyMin = 32

	maxPayloadBytes = 64 << 10
	maxBatch        = 100_000
	// MaxK bounds a single search so one request cannot ask for the whole index.
	MaxK      = 10_000
	defaultEf = 100
)

// CollectionConfig is fixed when a collection is created.
type CollectionConfig struct {
	Dim            int    `json:"dim"`
	Metric         string `json:"metric"`
	M              int    `json:"m"`
	EfConstruction int    `json:"ef_construction"`
}

// Point is a vector with an id and an optional JSON object payload.
type Point struct {
	ID      uint64          `json:"id"`
	Vector  []float32       `json:"vector"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// Hit is one search result. Distance follows the collection metric: smaller
// is closer.
type Hit struct {
	ID       uint64         `json:"id"`
	Distance float32        `json:"distance"`
	Payload  map[string]any `json:"payload,omitempty"`
	Vector   []float32      `json:"vector,omitempty"`
}

// SearchRequest describes one nearest-neighbour query.
type SearchRequest struct {
	Vector      []float32 `json:"vector"`
	K           int       `json:"k"`
	Ef          int       `json:"ef"`
	Filter      *Filter   `json:"filter"`
	WithPayload bool      `json:"with_payload"`
	WithVector  bool      `json:"with_vector"`
}

// Stats is a point-in-time view of a collection.
type Stats struct {
	Name        string           `json:"name"`
	Config      CollectionConfig `json:"config"`
	Count       int              `json:"count"`
	LastLSN     uint64           `json:"last_lsn"`
	SnapshotLSN uint64           `json:"snapshot_lsn"`
}

// Collection is one named index with its payloads and its own log.
//
// A write is appended to the WAL and then applied in memory. Two locks keep
// that sound without serialising all writers:
//
//   - stripes: a write holds the stripe of every id it touches across both
//     steps, so two writes to the same id reach the log and the index in the
//     same order. Writes to different ids commute and run concurrently,
//     which lets them share one fsync.
//   - ckpt: writers hold it shared; a snapshot holds it exclusively, so the
//     state it serialises is exactly the log prefix up to the LSN it records.
type Collection struct {
	name   string
	dir    string
	cfg    CollectionConfig
	metric vec.Metric

	index    *hnsw.Index
	payloads payloadStore
	log      *wal.Log

	stripes [numStripes]sync.Mutex
	ckpt    sync.RWMutex
	snapLSN atomic.Uint64
	closed  bool // guarded by ckpt
}

func (c *Collection) Name() string             { return c.name }
func (c *Collection) Config() CollectionConfig { return c.cfg }

func (c *Collection) Stats() Stats {
	return Stats{
		Name:        c.name,
		Config:      c.cfg,
		Count:       c.index.Len(),
		LastLSN:     c.log.LastLSN(),
		SnapshotLSN: c.snapLSN.Load(),
	}
}

func openCollection(dir, name string, cfg CollectionConfig, walOpts wal.Options) (*Collection, error) {
	metric, err := vec.ParseMetric(cfg.Metric)
	if err != nil {
		return nil, err
	}
	c := &Collection{name: name, dir: dir, cfg: cfg, metric: metric}
	if err := c.loadSnapshot(); err != nil {
		return nil, fmt.Errorf("collection %s: load snapshot: %w", name, err)
	}
	if c.index == nil {
		c.index = hnsw.New(cfg.Dim, hnsw.Config{M: cfg.M, EfConstruction: cfg.EfConstruction, Metric: metric})
	}
	log, err := wal.Open(filepath.Join(dir, "wal"), walOpts)
	if err != nil {
		return nil, fmt.Errorf("collection %s: open wal: %w", name, err)
	}
	snap := c.snapLSN.Load()
	if log.LastLSN() < snap {
		// The snapshot is ahead of the log, so the log directory was lost or
		// replaced. New records would reuse LSNs the snapshot already covers
		// and be skipped on the next start.
		log.Close()
		return nil, fmt.Errorf("collection %s: wal ends at lsn %d but snapshot is at %d", name, log.LastLSN(), snap)
	}
	err = log.Replay(snap+1, func(_ uint64, rec []byte) error {
		op, entries, err := decodeRecord(rec, cfg.Dim)
		if err != nil {
			return err
		}
		return c.apply(op, entries)
	})
	if err != nil {
		log.Close()
		return nil, fmt.Errorf("collection %s: replay wal: %w", name, err)
	}
	c.log = log
	return c, nil
}

// Upsert inserts the points, replacing any that already exist. The batch is
// one WAL record: after a crash either all of it is recovered or none.
func (c *Collection) Upsert(points []Point) error {
	entries, err := c.validate(points)
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return nil
	}
	rec := encodeUpsert(points, c.cfg.Dim)
	return c.write(rec, opUpsert, entries)
}

// Delete removes the given ids and reports how many were present.
func (c *Collection) Delete(ids []uint64) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	if len(ids) > maxBatch {
		return 0, fmt.Errorf("%w: batch of %d ids exceeds %d", ErrInvalid, len(ids), maxBatch)
	}
	entries := make([]entry, len(ids))
	for i, id := range ids {
		entries[i].id = id
	}
	before := c.index.Len()
	if err := c.write(encodeDelete(ids), opDelete, entries); err != nil {
		return 0, err
	}
	// Concurrent writers can move Len, so this is a best-effort count.
	return max(before-c.index.Len(), 0), nil
}

func (c *Collection) write(rec []byte, op byte, entries []entry) error {
	c.ckpt.RLock()
	defer c.ckpt.RUnlock()
	if c.closed {
		return ErrClosed
	}
	unlock := c.lockStripes(entries)
	defer unlock()
	if _, err := c.log.Append(rec); err != nil {
		return fmt.Errorf("wal append: %w", err)
	}
	return c.apply(op, entries)
}

// lockStripes takes the stripe of every id in ascending stripe order, which
// is what keeps two overlapping batches from deadlocking.
func (c *Collection) lockStripes(entries []entry) (unlock func()) {
	if len(entries) == 1 {
		mu := &c.stripes[stripe(entries[0].id)]
		mu.Lock()
		return mu.Unlock
	}
	var want [numStripes]bool
	for i := range entries {
		want[stripe(entries[i].id)] = true
	}
	for i := range want {
		if want[i] {
			c.stripes[i].Lock()
		}
	}
	return func() {
		for i := range want {
			if want[i] {
				c.stripes[i].Unlock()
			}
		}
	}
}

func stripe(id uint64) int {
	return int((id * 0x9E3779B97F4A7C15) >> 56)
}

// apply mutates the in-memory state. It is shared by the live write path and
// by WAL replay, so both produce the same index.
func (c *Collection) apply(op byte, entries []entry) error {
	one := func(e *entry) error {
		c.index.Delete(e.id)
		if op == opDelete {
			c.payloads.set(e.id, nil)
			return nil
		}
		if err := c.index.Add(e.id, e.vector); err != nil {
			return fmt.Errorf("index add %d: %w", e.id, err)
		}
		c.payloads.set(e.id, e.payload)
		return nil
	}
	workers := min(runtime.GOMAXPROCS(0), len(entries)/(parallelApplyMin/2))
	if op == opDelete || len(entries) < parallelApplyMin || workers < 2 {
		for i := range entries {
			if err := one(&entries[i]); err != nil {
				return err
			}
		}
		return nil
	}
	var (
		next     atomic.Int64
		wg       sync.WaitGroup
		errOnce  sync.Once
		firstErr error
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(entries) {
					return
				}
				if err := one(&entries[i]); err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
			}
		}()
	}
	wg.Wait()
	return firstErr
}

// validate checks a batch before anything is logged, so the WAL only ever
// holds records that apply cleanly.
func (c *Collection) validate(points []Point) ([]entry, error) {
	if len(points) > maxBatch {
		return nil, fmt.Errorf("%w: batch of %d points exceeds %d", ErrInvalid, len(points), maxBatch)
	}
	entries := make([]entry, len(points))
	seen := make(map[uint64]struct{}, len(points))
	for i := range points {
		p := &points[i]
		if _, dup := seen[p.ID]; dup {
			return nil, fmt.Errorf("%w: id %d appears twice in one batch", ErrInvalid, p.ID)
		}
		seen[p.ID] = struct{}{}
		if err := c.checkVector(p.Vector); err != nil {
			return nil, fmt.Errorf("point %d: %w", p.ID, err)
		}
		var payload map[string]any
		if len(p.Payload) > 0 {
			if len(p.Payload) > maxPayloadBytes {
				return nil, fmt.Errorf("%w: point %d payload exceeds %d bytes", ErrInvalid, p.ID, maxPayloadBytes)
			}
			if err := json.Unmarshal(p.Payload, &payload); err != nil {
				return nil, fmt.Errorf("%w: point %d payload must be a JSON object", ErrInvalid, p.ID)
			}
			if payload == nil {
				// A literal null: store it as no payload.
				p.Payload = nil
			}
		}
		entries[i] = entry{id: p.ID, vector: p.Vector, payload: payload}
	}
	return entries, nil
}

func (c *Collection) checkVector(v []float32) error {
	if len(v) != c.cfg.Dim {
		return fmt.Errorf("%w: vector has %d dimensions, collection has %d", ErrInvalid, len(v), c.cfg.Dim)
	}
	var norm float64
	for _, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%w: vector contains NaN or Inf", ErrInvalid)
		}
		norm += f * f
	}
	if c.metric == vec.Cosine && norm == 0 {
		return fmt.Errorf("%w: zero vector has no direction under the cosine metric", ErrInvalid)
	}
	return nil
}

// Search returns the nearest points to req.Vector, closest first.
func (c *Collection) Search(req SearchRequest) ([]Hit, error) {
	if err := c.checkVector(req.Vector); err != nil {
		return nil, err
	}
	if req.K <= 0 {
		req.K = 10
	}
	if req.K > MaxK {
		return nil, fmt.Errorf("%w: k is capped at %d", ErrInvalid, MaxK)
	}
	if req.Ef <= 0 {
		req.Ef = defaultEf
	}
	if err := req.Filter.Validate(); err != nil {
		return nil, err
	}
	var allow func(uint64) bool
	if f := req.Filter; f != nil {
		allow = func(id uint64) bool { return f.Matches(c.payloads.get(id)) }
	}
	res := c.index.Search(req.Vector, req.K, req.Ef, allow)
	hits := make([]Hit, len(res))
	for i, r := range res {
		hits[i] = Hit{ID: r.ID, Distance: r.Dist}
		if req.WithPayload {
			hits[i].Payload = c.payloads.get(r.ID)
		}
		if req.WithVector {
			hits[i].Vector, _ = c.index.Vector(r.ID)
		}
	}
	return hits, nil
}

// Get returns one point by id.
func (c *Collection) Get(id uint64) (Hit, error) {
	v, ok := c.index.Vector(id)
	if !ok {
		return Hit{}, fmt.Errorf("%w: point %d", ErrNotFound, id)
	}
	return Hit{ID: id, Vector: v, Payload: c.payloads.get(id)}, nil
}

// Close flushes and closes the log. It does not snapshot.
func (c *Collection) Close() error {
	c.ckpt.Lock()
	defer c.ckpt.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return c.log.Close()
}

func sortedIDs(m map[string]*Collection) []string {
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
