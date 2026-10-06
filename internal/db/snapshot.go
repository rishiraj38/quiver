package db

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"

	"github.com/rishiraj38/quiver/internal/hnsw"
)

// Snapshot file layout, little-endian:
//
//	magic "QVSNAP01" | lsn u64 | payload count u64
//	count * (id u64 | len u32 | JSON) | crc32 of everything so far
//	hnsw index stream (carries its own checksum)
const (
	snapshotName  = "snapshot.bin"
	snapshotMagic = "QVSNAP01"

	maxSnapshotPayload = 16 << 20
)

var errBadSnapshot = errors.New("db: corrupt snapshot")

// Snapshot writes the whole collection to disk and drops the WAL segments it
// makes redundant. Writers are paused for the duration; searches are not.
//
// The file is written under a temporary name and renamed into place, so a
// crash at any point leaves either the old snapshot or the new one.
func (c *Collection) Snapshot() error {
	c.ckpt.Lock()
	defer c.ckpt.Unlock()
	if c.closed {
		return ErrClosed
	}
	lsn := c.log.LastLSN()
	if lsn == c.snapLSN.Load() {
		return nil
	}
	final := filepath.Join(c.dir, snapshotName)
	tmp := final + ".tmp"
	if err := c.writeSnapshot(tmp, lsn); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := syncDir(c.dir); err != nil {
		return err
	}
	c.snapLSN.Store(lsn)
	return c.log.TruncateBefore(lsn + 1)
}

func (c *Collection) writeSnapshot(path string, lsn uint64) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	bw := bufio.NewWriterSize(f, 1<<20)
	sum := crc32.NewIEEE()
	head := io.MultiWriter(bw, sum)

	var scratch [20]byte
	copy(scratch[:8], snapshotMagic)
	binary.LittleEndian.PutUint64(scratch[8:], lsn)
	if _, err := head.Write(scratch[:16]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(scratch[:], uint64(c.payloads.len()))
	if _, err := head.Write(scratch[:8]); err != nil {
		return err
	}
	err = c.payloads.each(func(id uint64, p map[string]any) error {
		raw, err := json.Marshal(p)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(scratch[:], id)
		binary.LittleEndian.PutUint32(scratch[8:], uint32(len(raw)))
		if _, err := head.Write(scratch[:12]); err != nil {
			return err
		}
		_, err = head.Write(raw)
		return err
	})
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(scratch[:], sum.Sum32())
	if _, err := bw.Write(scratch[:4]); err != nil {
		return err
	}
	if _, err := c.index.WriteTo(bw); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return f.Sync()
}

// loadSnapshot restores index, payloads and snapLSN. A missing file is not an
// error: the collection is rebuilt from the log alone.
func (c *Collection) loadSnapshot() error {
	f, err := os.Open(filepath.Join(c.dir, snapshotName))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	sum := crc32.NewIEEE()
	head := io.TeeReader(br, sum)

	var scratch [16]byte
	if _, err := io.ReadFull(head, scratch[:16]); err != nil {
		return errBadSnapshot
	}
	if string(scratch[:8]) != snapshotMagic {
		return errBadSnapshot
	}
	lsn := binary.LittleEndian.Uint64(scratch[8:])
	if _, err := io.ReadFull(head, scratch[:8]); err != nil {
		return errBadSnapshot
	}
	count := binary.LittleEndian.Uint64(scratch[:8])
	var raw []byte
	for range count {
		if _, err := io.ReadFull(head, scratch[:12]); err != nil {
			return errBadSnapshot
		}
		id := binary.LittleEndian.Uint64(scratch[:8])
		n := int(binary.LittleEndian.Uint32(scratch[8:12]))
		// Re-marshalled JSON can be larger than what the client sent, so
		// this is only a guard against a garbage length.
		if n > maxSnapshotPayload {
			return errBadSnapshot
		}
		if cap(raw) < n {
			raw = make([]byte, n)
		}
		raw = raw[:n]
		if _, err := io.ReadFull(head, raw); err != nil {
			return errBadSnapshot
		}
		var p map[string]any
		if err := json.Unmarshal(raw, &p); err != nil {
			return errBadSnapshot
		}
		c.payloads.set(id, p)
	}
	want := sum.Sum32()
	if _, err := io.ReadFull(br, scratch[:4]); err != nil {
		return errBadSnapshot
	}
	if binary.LittleEndian.Uint32(scratch[:4]) != want {
		return errBadSnapshot
	}
	index, err := hnsw.Load(br)
	if err != nil {
		return err
	}
	if index.Dim() != c.cfg.Dim {
		return fmt.Errorf("snapshot has dimension %d, collection has %d", index.Dim(), c.cfg.Dim)
	}
	c.index = index
	c.snapLSN.Store(lsn)
	return nil
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
