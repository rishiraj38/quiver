// Package wal is a segmented, checksummed write-ahead log.
package wal

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// SyncPolicy decides when appended records reach stable storage.
type SyncPolicy uint8

const (
	SyncAlways   SyncPolicy = iota // fsync before Append returns; concurrent appends share one fsync (group commit)
	SyncInterval                   // fsync in the background every Options.Interval
	SyncNever                      // leave it to the OS
)

type Options struct {
	Sync        SyncPolicy
	Interval    time.Duration // for SyncInterval; default 100ms
	SegmentSize int64         // roll to a new segment file past this many bytes; default 64 MiB
}

// ErrCorrupt reports a bad record that is not a torn tail, i.e. valid
// records follow it or it sits in a sealed segment.
var ErrCorrupt = errors.New("wal: corrupt record")

// ErrClosed is returned by operations on a closed Log.
var ErrClosed = errors.New("wal: log is closed")

// ErrTooLarge is returned by Append for a payload over MaxRecordSize.
var ErrTooLarge = errors.New("wal: record exceeds MaxRecordSize")

// Log is safe for concurrent use.
//
// Lock order is retire, then mu, then syncMu. Nothing takes mu while
// holding syncMu, which is what lets appends continue during an fsync.
type Log struct {
	dir  string
	opts Options

	// mu serialises appends, so file order and LSN order are the same thing.
	mu         sync.Mutex
	segments   []uint64 // first LSN of each segment, ascending; the last is the active one
	activeSize int64
	frame      []byte // scratch for building a record so it goes out in one write
	closed     bool

	// syncMu is held across every fsync and is the whole of group commit:
	// appenders that arrive during an fsync queue here, the first one out
	// syncs everything written so far, and the rest find their record
	// already covered.
	syncMu  sync.Mutex
	durable uint64 // every record with LSN <= durable has been fsynced
	fsyncs  uint64

	// active changes only with both mu and syncMu held, so holding either
	// is enough to read it.
	active *os.File

	// lastLSN is stored after the record's write returns, so a syncer that
	// loads it and then fsyncs is certain to cover that record. It is an
	// atomic because the syncer reads it without mu.
	lastLSN atomic.Uint64

	failure atomic.Pointer[error]

	// retire keeps TruncateBefore from unlinking a segment that a Replay
	// has listed but not opened yet.
	retire sync.RWMutex

	stop chan struct{}
	done chan struct{}
}

// Open opens or creates the log in dir. A torn record at the very end of the
// last segment (a crash mid-write) is truncated away; corruption anywhere
// else returns ErrCorrupt.
func Open(dir string, opts Options) (*Log, error) {
	if opts.Interval <= 0 {
		opts.Interval = 100 * time.Millisecond
	}
	if opts.SegmentSize <= 0 {
		opts.SegmentSize = 64 << 20
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	firsts, err := listSegments(dir)
	if err != nil {
		return nil, err
	}

	l := &Log{dir: dir, opts: opts}
	if len(firsts) == 0 {
		if l.active, err = createSegment(dir, 1); err != nil {
			return nil, err
		}
		l.segments = []uint64{1}
		l.activeSize = headerSize
	} else {
		next := firsts[0]
		for i, first := range firsts {
			// Each segment must start where the previous one ended. A gap
			// means a segment file is missing or a sealed one lost records.
			if first != next {
				return nil, fmt.Errorf("%w: segment %s should start at LSN %d", ErrCorrupt, segmentName(first), next)
			}
			path := filepath.Join(dir, segmentName(first))
			var count uint64
			if i < len(firsts)-1 {
				count, err = scanSealed(path)
			} else {
				l.active, l.activeSize, count, err = recoverActive(path)
			}
			if err != nil {
				return nil, err
			}
			next += count
		}
		l.segments = firsts
		l.lastLSN.Store(next - 1)
		l.durable = next - 1
	}

	if opts.Sync == SyncInterval {
		l.stop = make(chan struct{})
		l.done = make(chan struct{})
		go l.syncLoop()
	}
	return l, nil
}

// Append writes one record and returns its log sequence number. LSNs start
// at 1 and increase by one per record, across segments and restarts.
//
// If Append returns an error the record's fate is unknown: it may or may not
// be in the log after a restart.
func (l *Log) Append(payload []byte) (lsn uint64, err error) {
	if len(payload) > MaxRecordSize {
		return 0, ErrTooLarge
	}
	l.mu.Lock()
	lsn, err = l.appendLocked(payload)
	l.mu.Unlock()
	if err != nil {
		return 0, err
	}
	if l.opts.Sync == SyncAlways {
		if err := l.syncThrough(lsn); err != nil {
			return 0, err
		}
	}
	return lsn, nil
}

func (l *Log) appendLocked(payload []byte) (uint64, error) {
	if l.closed {
		return 0, ErrClosed
	}
	if err := l.failed(); err != nil {
		return 0, err
	}
	lsn := l.lastLSN.Load() + 1
	need := int64(frameHeaderSize + len(payload))
	// Never roll an empty segment: a record larger than SegmentSize gets a
	// segment to itself instead of rolling forever.
	if l.activeSize > headerSize && l.activeSize+need > l.opts.SegmentSize {
		if err := l.roll(lsn); err != nil {
			return 0, err
		}
	}
	l.frame = appendFrame(l.frame[:0], payload)
	_, err := l.active.WriteAt(l.frame, l.activeSize)
	if cap(l.frame) > 1<<20 {
		l.frame = nil
	}
	if err != nil {
		// A failed write may have left part of a record on disk. Appending
		// after it would bury a bad record under good ones, which recovery
		// reads as corruption, so stop here and let reopen trim the tail.
		return 0, l.fail(err)
	}
	l.activeSize += need
	l.lastLSN.Store(lsn)
	return lsn, nil
}

// roll seals the active segment and starts a new one whose first record
// will be next. Called with mu held.
func (l *Log) roll(next uint64) error {
	l.syncMu.Lock()
	defer l.syncMu.Unlock()
	// Sealing fsyncs under every policy, including SyncNever. Recovery
	// depends on it: only the last segment may have a torn tail.
	if err := l.syncLocked(); err != nil {
		return err
	}
	f, err := createSegment(l.dir, next)
	if err != nil {
		return err
	}
	old := l.active
	l.active = f
	l.activeSize = headerSize
	l.segments = append(l.segments, next)
	old.Close() // already fsynced; nothing a close error could still lose
	return nil
}

// syncThrough returns once the record at lsn is durable.
func (l *Log) syncThrough(lsn uint64) error {
	l.syncMu.Lock()
	defer l.syncMu.Unlock()
	if l.durable >= lsn {
		return nil // an fsync that started after our write already covered it
	}
	return l.syncLocked()
}

// syncLocked fsyncs the active segment if it holds anything not yet
// durable. Called with syncMu held.
func (l *Log) syncLocked() error {
	if err := l.failed(); err != nil {
		return err
	}
	if l.active == nil {
		return ErrClosed
	}
	// Loaded before the fsync starts, so every record up to here is already
	// in the file. Earlier segments were synced when they were sealed.
	upto := l.lastLSN.Load()
	if upto == l.durable {
		return nil
	}
	l.fsyncs++
	if err := l.active.Sync(); err != nil {
		// After a failed fsync the kernel may have dropped the dirty pages
		// and a retry can report success without the data being on disk.
		// The only honest answer is to stop accepting writes.
		return l.fail(err)
	}
	l.durable = upto
	return nil
}

func (l *Log) syncLoop() {
	defer close(l.done)
	t := time.NewTicker(l.opts.Interval)
	defer t.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-t.C:
			l.syncMu.Lock()
			l.syncLocked() // a failure is sticky and surfaces on the next Append
			l.syncMu.Unlock()
		}
	}
}

// fail records the first I/O error and returns the error callers should see
// from then on.
func (l *Log) fail(err error) error {
	err = fmt.Errorf("wal: log failed and must be reopened: %w", err)
	l.failure.CompareAndSwap(nil, &err)
	return *l.failure.Load()
}

func (l *Log) failed() error {
	if p := l.failure.Load(); p != nil {
		return *p
	}
	return nil
}

// Replay calls fn for every record with LSN >= from, in order. The payload
// slice is only valid during the call.
//
// Replay reads from the segment files one record at a time and may run
// while appends continue; it covers every record appended before it was
// called. fn may call Append but must not call TruncateBefore.
func (l *Log) Replay(from uint64, fn func(lsn uint64, payload []byte) error) error {
	l.retire.RLock()
	defer l.retire.RUnlock()

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	segments := slices.Clone(l.segments)
	activeSize := l.activeSize
	end := l.lastLSN.Load() + 1
	l.mu.Unlock()

	for i, first := range segments {
		last := i == len(segments)-1
		limit := end
		if !last {
			limit = segments[i+1]
		}
		if limit <= from {
			continue
		}
		// The active segment is read only up to the size captured above. A
		// record being appended right now may be half written past that
		// point, and it is not part of what this call promised to deliver.
		size := int64(-1)
		if last {
			size = activeSize
		}
		if err := l.replaySegment(first, limit, size, from, fn); err != nil {
			return err
		}
	}
	return nil
}

// replaySegment delivers the records of one segment, whose LSNs are
// [first, limit). A negative size means read the whole file.
func (l *Log) replaySegment(first, limit uint64, size int64, from uint64, fn func(uint64, []byte) error) error {
	name := segmentName(first)
	f, err := os.Open(filepath.Join(l.dir, name))
	if err != nil {
		return err
	}
	defer f.Close()
	if size < 0 {
		st, err := f.Stat()
		if err != nil {
			return err
		}
		size = st.Size()
	}
	if err := readHeader(f); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	r := newRecordReader(f, size)
	lsn := first
	for {
		payload, err := r.next(lsn < from)
		if err == io.EOF {
			break
		}
		if err == errBadRecord {
			return fmt.Errorf("%w: segment %s, offset %d", ErrCorrupt, name, r.offset)
		}
		if err != nil {
			return err
		}
		if lsn >= from {
			if err := fn(lsn, payload); err != nil {
				return err
			}
		}
		lsn++
	}
	if lsn != limit {
		return fmt.Errorf("%w: segment %s ends at LSN %d, expected %d", ErrCorrupt, name, lsn-1, limit-1)
	}
	return nil
}

// LastLSN is the LSN of the newest record, or 0 for an empty log.
func (l *Log) LastLSN() uint64 { return l.lastLSN.Load() }

// TruncateBefore deletes whole segments whose records all have LSN < lsn.
// It is called after a snapshot makes those records redundant.
//
// The active segment is never deleted, so the log keeps its position even
// if lsn is past the end.
func (l *Log) TruncateBefore(lsn uint64) error {
	l.retire.Lock()
	defer l.retire.Unlock()

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	// Segment i ends just before segment i+1 starts, so it is entirely
	// below lsn when the next segment's first LSN is <= lsn.
	n := 0
	for n < len(l.segments)-1 && l.segments[n+1] <= lsn {
		n++
	}
	victims := slices.Clone(l.segments[:n])
	l.mu.Unlock()
	if n == 0 {
		return nil
	}

	// Oldest first and stop at the first failure, so whatever is left on
	// disk is still one contiguous run of LSNs that Open will accept.
	// Appends are not blocked meanwhile: they only add to the far end of
	// the list, and retire keeps other readers of the front away.
	removed := 0
	var err error
	for _, first := range victims {
		if err = os.Remove(filepath.Join(l.dir, segmentName(first))); err != nil {
			break
		}
		removed++
	}
	l.mu.Lock()
	l.segments = slices.Delete(l.segments, 0, removed)
	l.mu.Unlock()
	if removed > 0 {
		if serr := syncDir(l.dir); err == nil {
			err = serr
		}
	}
	return err
}

// Sync forces buffered records to disk.
func (l *Log) Sync() error {
	l.syncMu.Lock()
	defer l.syncMu.Unlock()
	return l.syncLocked()
}

// Close syncs and closes the log. Calling it again is a no-op.
func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.stop != nil {
		// The sync loop only ever takes syncMu, so waiting for it while
		// holding mu cannot deadlock.
		close(l.stop)
		<-l.done
	}
	l.syncMu.Lock()
	defer l.syncMu.Unlock()
	// A clean shutdown is durable under every policy.
	err := l.syncLocked()
	if cerr := l.active.Close(); err == nil {
		err = cerr
	}
	l.active = nil
	return err
}
