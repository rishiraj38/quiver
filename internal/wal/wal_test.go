package wal

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// smallSegments makes a log roll every few records.
var smallSegments = Options{Sync: SyncNever, SegmentSize: 256}

func mustOpen(t testing.TB, dir string, opts Options) *Log {
	t.Helper()
	l, err := Open(dir, opts)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { l.Close() })
	return l
}

func payloadFor(i int) []byte {
	return []byte(fmt.Sprintf("record-%04d-%s", i, bytes.Repeat([]byte{byte('a' + i%26)}, i%40)))
}

// appendN appends records numbered from..to inclusive and checks each LSN
// equals its number.
func appendN(t testing.TB, l *Log, from, to int) {
	t.Helper()
	for i := from; i <= to; i++ {
		lsn, err := l.Append(payloadFor(i))
		if err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
		if lsn != uint64(i) {
			t.Fatalf("Append %d returned LSN %d", i, lsn)
		}
	}
}

// checkReplay asserts that Replay(from) yields exactly records from..to
// with the payloads appendN wrote.
func checkReplay(t testing.TB, l *Log, from, to int) {
	t.Helper()
	want := from
	err := l.Replay(uint64(from), func(lsn uint64, p []byte) error {
		if lsn != uint64(want) {
			t.Fatalf("replay: got LSN %d, want %d", lsn, want)
		}
		if !bytes.Equal(p, payloadFor(want)) {
			t.Fatalf("replay: LSN %d payload %q, want %q", lsn, p, payloadFor(want))
		}
		want++
		return nil
	})
	if err != nil {
		t.Fatalf("Replay(%d): %v", from, err)
	}
	if want != to+1 {
		t.Fatalf("Replay(%d) stopped at LSN %d, want %d", from, want-1, to)
	}
}

func segmentFiles(t testing.TB, dir string) []string {
	t.Helper()
	firsts, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	paths := make([]string, len(firsts))
	for i, first := range firsts {
		paths[i] = filepath.Join(dir, segmentName(first))
	}
	return paths
}

// buildLog writes n records with small segments, closes the log, and
// returns the directory.
func buildLog(t testing.TB, n int) string {
	t.Helper()
	dir := t.TempDir()
	l := mustOpen(t, dir, smallSegments)
	appendN(t, l, 1, n)
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	return dir
}

// cloneDir copies every file in src into a fresh temp directory.
func cloneDir(t testing.TB, src string) string {
	t.Helper()
	dst := t.TempDir()
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

// recordOffsets returns the file offset of each record in a segment, plus
// the offset just past the last one.
func recordOffsets(t testing.TB, path string) []int64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	r := newRecordReader(f, st.Size())
	offs := []int64{r.offset}
	for {
		if _, err := r.next(false); err != nil {
			break
		}
		offs = append(offs, r.offset)
	}
	return offs
}

func TestRoundTripAcrossSegments(t *testing.T) {
	dir := t.TempDir()
	l := mustOpen(t, dir, smallSegments)
	if got := l.LastLSN(); got != 0 {
		t.Fatalf("LastLSN of empty log = %d", got)
	}
	checkReplay(t, l, 1, 0)

	const n = 200
	appendN(t, l, 1, n)
	if got := l.LastLSN(); got != n {
		t.Fatalf("LastLSN = %d, want %d", got, n)
	}
	files := segmentFiles(t, dir)
	if len(files) < 5 {
		t.Fatalf("expected several segments, got %d", len(files))
	}
	for _, p := range files[:len(files)-1] {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if st.Size() > smallSegments.SegmentSize {
			t.Fatalf("%s is %d bytes, over SegmentSize", filepath.Base(p), st.Size())
		}
	}
	checkReplay(t, l, 1, n)
}

func TestReopenContinuesLSNs(t *testing.T) {
	dir := t.TempDir()
	next := 1
	for round := 0; round < 4; round++ {
		l := mustOpen(t, dir, smallSegments)
		if got := l.LastLSN(); got != uint64(next-1) {
			t.Fatalf("round %d: LastLSN = %d, want %d", round, got, next-1)
		}
		appendN(t, l, next, next+36)
		next += 37
		checkReplay(t, l, 1, next-1)
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestReplayFromMiddle(t *testing.T) {
	dir := buildLog(t, 120)
	l := mustOpen(t, dir, smallSegments)
	// Every starting point, so segment boundaries and mid-segment skips are
	// both covered, plus one past the end.
	for from := 1; from <= 121; from++ {
		checkReplay(t, l, from, 120)
	}
	// From 0 is the same as from 1: there is no record 0.
	n := 0
	if err := l.Replay(0, func(uint64, []byte) error { n++; return nil }); err != nil || n != 120 {
		t.Fatalf("Replay(0): %d records, err %v", n, err)
	}
}

func TestReplayStopsOnCallbackError(t *testing.T) {
	dir := buildLog(t, 50)
	l := mustOpen(t, dir, smallSegments)
	stop := errors.New("stop")
	var seen uint64
	err := l.Replay(1, func(lsn uint64, _ []byte) error {
		seen = lsn
		if lsn == 17 {
			return stop
		}
		return nil
	})
	if err != stop || seen != 17 {
		t.Fatalf("got err %v at LSN %d", err, seen)
	}
}

func TestEmptyAndOversizedPayloads(t *testing.T) {
	dir := t.TempDir()
	l := mustOpen(t, dir, smallSegments)
	big := bytes.Repeat([]byte("x"), 10*int(smallSegments.SegmentSize))
	want := [][]byte{nil, big, []byte("after"), {}}
	for _, p := range want {
		if _, err := l.Append(p); err != nil {
			t.Fatal(err)
		}
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}

	l = mustOpen(t, dir, smallSegments)
	i := 0
	err := l.Replay(1, func(lsn uint64, p []byte) error {
		if !bytes.Equal(p, want[i]) {
			t.Fatalf("LSN %d: got %d bytes, want %d", lsn, len(p), len(want[i]))
		}
		i++
		return nil
	})
	if err != nil || i != len(want) {
		t.Fatalf("replayed %d records, err %v", i, err)
	}

	if _, err := l.Append(make([]byte, MaxRecordSize+1)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("oversized Append: %v", err)
	}
	if got := l.LastLSN(); got != uint64(len(want)) {
		t.Fatalf("rejected append moved LastLSN to %d", got)
	}
}

// A crash can cut the last record at any byte. Whatever the cut, the
// records before it survive, the partial one is gone from the file, and the
// log carries on from there.
func TestTornTailEveryOffset(t *testing.T) {
	const n = 30
	src := buildLog(t, n)
	files := segmentFiles(t, src)
	if len(files) < 3 {
		t.Fatalf("want a multi-segment log, got %d segments", len(files))
	}
	lastName := filepath.Base(files[len(files)-1])
	offs := recordOffsets(t, files[len(files)-1])
	if len(offs) < 3 {
		t.Fatalf("want at least two records in the last segment, got %d", len(offs)-1)
	}
	start, end := offs[len(offs)-2], offs[len(offs)-1]

	for cut := start; cut < end; cut++ {
		dir := cloneDir(t, src)
		path := filepath.Join(dir, lastName)
		if err := os.Truncate(path, cut); err != nil {
			t.Fatal(err)
		}
		l, err := Open(dir, smallSegments)
		if err != nil {
			t.Fatalf("cut at %d: Open: %v", cut, err)
		}
		if got := l.LastLSN(); got != n-1 {
			t.Fatalf("cut at %d: LastLSN = %d, want %d", cut, got, n-1)
		}
		checkReplay(t, l, 1, n-1)
		if st, err := os.Stat(path); err != nil || st.Size() != start {
			t.Fatalf("cut at %d: file is %d bytes after recovery, want %d (err %v)", cut, st.Size(), start, err)
		}
		appendN(t, l, n, n+5)
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}

		l, err = Open(dir, smallSegments)
		if err != nil {
			t.Fatalf("cut at %d: reopen: %v", cut, err)
		}
		checkReplay(t, l, 1, n+5)
		l.Close()
	}
}

// A tear does not have to shorten the file: the size can be extended before
// the data blocks land, leaving zeros or stale bytes where the record
// should be.
func TestTornTailGarbage(t *testing.T) {
	const n = 30
	src := buildLog(t, n)
	files := segmentFiles(t, src)
	lastName := filepath.Base(files[len(files)-1])
	offs := recordOffsets(t, files[len(files)-1])
	start, end := offs[len(offs)-2], offs[len(offs)-1]

	cases := map[string]func(b []byte) []byte{
		"zeroed record":       func(b []byte) []byte { clear(b[start:end]); return b },
		"zeroed payload tail": func(b []byte) []byte { clear(b[end-3 : end]); return b },
		"zeros past the end":  func(b []byte) []byte { return append(b, make([]byte, 4096)...) },
		"noise past the end":  func(b []byte) []byte { return append(b, bytes.Repeat([]byte{0xA5, 0x01, 0x00}, 500)...) },
		"flipped last record": func(b []byte) []byte { b[end-1] ^= 0xFF; return b },
	}
	for name, damage := range cases {
		t.Run(name, func(t *testing.T) {
			dir := cloneDir(t, src)
			path := filepath.Join(dir, lastName)
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			grown := len(b)
			b = damage(b)
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			survivors := n - 1
			if len(b) > grown {
				survivors = n // the damage is after the last record
			}
			l := mustOpen(t, dir, smallSegments)
			checkReplay(t, l, 1, survivors)
			appendN(t, l, survivors+1, survivors+3)
			l.Close()
			l = mustOpen(t, dir, smallSegments)
			checkReplay(t, l, 1, survivors+3)
		})
	}
}

// A crash while a segment is being created leaves a last segment with a
// missing or partial header.
func TestTornSegmentHeader(t *testing.T) {
	const n = 30
	src := buildLog(t, n)
	files := segmentFiles(t, src)
	prev := recordOffsets(t, files[len(files)-2])
	lastName := filepath.Base(files[len(files)-1])
	first, _ := parseSegmentName(lastName)
	survivors := int(first) - 1
	if len(prev) < 2 || survivors < 1 {
		t.Fatal("test log too small")
	}

	for size := int64(0); size <= headerSize; size++ {
		dir := cloneDir(t, src)
		if err := os.Truncate(filepath.Join(dir, lastName), size); err != nil {
			t.Fatal(err)
		}
		l, err := Open(dir, smallSegments)
		if err != nil {
			t.Fatalf("header cut to %d bytes: Open: %v", size, err)
		}
		if got := l.LastLSN(); got != uint64(survivors) {
			t.Fatalf("header cut to %d bytes: LastLSN = %d, want %d", size, got, survivors)
		}
		appendN(t, l, survivors+1, survivors+10)
		l.Close()
		l = mustOpen(t, dir, smallSegments)
		checkReplay(t, l, 1, survivors+10)
		l.Close()
	}

	t.Run("zeroed header", func(t *testing.T) {
		dir := cloneDir(t, src)
		if err := os.WriteFile(filepath.Join(dir, lastName), make([]byte, headerSize), 0o644); err != nil {
			t.Fatal(err)
		}
		l := mustOpen(t, dir, smallSegments)
		appendN(t, l, survivors+1, survivors+2)
		checkReplay(t, l, 1, survivors+2)
	})

	t.Run("bad header with records is corruption", func(t *testing.T) {
		dir := cloneDir(t, src)
		flipByte(t, filepath.Join(dir, lastName), 1, 0xFF)
		if _, err := Open(dir, smallSegments); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("got %v, want ErrCorrupt", err)
		}
	})
}

func flipByte(t testing.TB, path string, off int64, mask byte) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b[off] ^= mask
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCorruptSealedSegment(t *testing.T) {
	src := buildLog(t, 60)
	files := segmentFiles(t, src)
	if len(files) < 4 {
		t.Fatalf("want several segments, got %d", len(files))
	}
	// Includes the last record of a sealed segment: being at the end of
	// its file does not make it a tail.
	for _, idx := range []int{0, 1, len(files) - 2} {
		name := filepath.Base(files[idx])
		st, err := os.Stat(files[idx])
		if err != nil {
			t.Fatal(err)
		}
		for off := int64(0); off < st.Size(); off++ {
			dir := cloneDir(t, src)
			flipByte(t, filepath.Join(dir, name), off, 0x40)
			l, err := Open(dir, smallSegments)
			if !errors.Is(err, ErrCorrupt) {
				if l != nil {
					l.Close()
				}
				// A flip in the version field is reported as an unsupported
				// version, which is still a refusal to open.
				if err != nil && off >= 4 && off < headerSize {
					continue
				}
				t.Fatalf("segment %s, byte %d flipped: got %v, want ErrCorrupt", name, off, err)
			}
		}
	}

	t.Run("truncated sealed segment", func(t *testing.T) {
		dir := cloneDir(t, src)
		name := filepath.Base(files[1])
		st, _ := os.Stat(files[1])
		if err := os.Truncate(filepath.Join(dir, name), st.Size()-3); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(dir, smallSegments); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("got %v, want ErrCorrupt", err)
		}
	})

	t.Run("missing segment", func(t *testing.T) {
		dir := cloneDir(t, src)
		if err := os.Remove(filepath.Join(dir, filepath.Base(files[1]))); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(dir, smallSegments); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("got %v, want ErrCorrupt", err)
		}
	})
}

// Damage to a record in the last segment that has good records after it
// must not be mistaken for a torn tail, whichever field it hits. A flip in
// the length field is the hard case: the next record boundary is lost.
func TestCorruptEarlyRecordInLastSegment(t *testing.T) {
	dir := t.TempDir()
	l := mustOpen(t, dir, Options{Sync: SyncNever})
	appendN(t, l, 1, 12)
	l.Close()
	files := segmentFiles(t, dir)
	if len(files) != 1 {
		t.Fatalf("want one segment, got %d", len(files))
	}
	name := filepath.Base(files[0])
	offs := recordOffsets(t, files[0])

	for _, rec := range []int{0, 5, 10} {
		for off := offs[rec]; off < offs[rec+1]; off++ {
			for _, mask := range []byte{0x01, 0x80, 0xFF} {
				clone := cloneDir(t, dir)
				flipByte(t, filepath.Join(clone, name), off, mask)
				l, err := Open(clone, Options{Sync: SyncNever})
				if !errors.Is(err, ErrCorrupt) {
					if l != nil {
						l.Close()
					}
					t.Fatalf("record %d, byte %d ^ %#x: got %v, want ErrCorrupt", rec+1, off-offs[rec], mask, err)
				}
			}
		}
	}
}

func TestTruncateBefore(t *testing.T) {
	const n = 200
	dir := buildLog(t, n)
	l := mustOpen(t, dir, smallSegments)
	firsts, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(firsts) < 6 {
		t.Fatalf("want several segments, got %d", len(firsts))
	}

	if err := l.TruncateBefore(0); err != nil {
		t.Fatal(err)
	}
	if err := l.TruncateBefore(firsts[0]); err != nil {
		t.Fatal(err)
	}
	if got, _ := listSegments(dir); len(got) != len(firsts) {
		t.Fatalf("nothing should be removed yet, have %d of %d segments", len(got), len(firsts))
	}

	// One below a boundary: the segment that ends there still holds a
	// record that is not below the LSN, so it stays.
	if err := l.TruncateBefore(firsts[3] - 1); err != nil {
		t.Fatal(err)
	}
	got, _ := listSegments(dir)
	if len(got) != len(firsts)-2 || got[0] != firsts[2] {
		t.Fatalf("after TruncateBefore(%d): first segment %d, %d left", firsts[3]-1, got[0], len(got))
	}
	checkReplay(t, l, int(firsts[2]), n)

	if err := l.TruncateBefore(firsts[3]); err != nil {
		t.Fatal(err)
	}
	got, _ = listSegments(dir)
	if got[0] != firsts[3] {
		t.Fatalf("after TruncateBefore(%d): first segment %d", firsts[3], got[0])
	}

	// Past the end: everything sealed goes, the active segment stays.
	if err := l.TruncateBefore(n + 1000); err != nil {
		t.Fatal(err)
	}
	got, _ = listSegments(dir)
	if len(got) != 1 || got[0] != firsts[len(firsts)-1] {
		t.Fatalf("active segment not preserved: %v", got)
	}
	if l.LastLSN() != n {
		t.Fatalf("LastLSN = %d after truncation", l.LastLSN())
	}
	appendN(t, l, n+1, n+40)
	checkReplay(t, l, int(got[0]), n+40)
	l.Close()

	// A log whose first segment no longer starts at 1 reopens normally.
	l = mustOpen(t, dir, smallSegments)
	if l.LastLSN() != n+40 {
		t.Fatalf("LastLSN after reopen = %d", l.LastLSN())
	}
	checkReplay(t, l, int(got[0]), n+40)
	appendN(t, l, n+41, n+42)
}

func TestConcurrentAppend(t *testing.T) {
	policies := map[string]Options{
		"always":   {Sync: SyncAlways, SegmentSize: 4096},
		"interval": {Sync: SyncInterval, Interval: 2 * time.Millisecond, SegmentSize: 4096},
		"never":    {Sync: SyncNever, SegmentSize: 4096},
	}
	for name, opts := range policies {
		t.Run(name, func(t *testing.T) {
			const writers, each = 16, 60
			dir := t.TempDir()
			l := mustOpen(t, dir, opts)

			var wg sync.WaitGroup
			var mu sync.Mutex
			byLSN := make(map[uint64]string)
			for w := 0; w < writers; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < each; i++ {
						p := fmt.Sprintf("writer-%02d-record-%03d", w, i)
						lsn, err := l.Append([]byte(p))
						if err != nil {
							t.Errorf("Append: %v", err)
							return
						}
						if opts.Sync == SyncAlways {
							l.syncMu.Lock()
							durable := l.durable
							l.syncMu.Unlock()
							if durable < lsn {
								t.Errorf("Append returned LSN %d with only %d durable", lsn, durable)
							}
						}
						mu.Lock()
						if prev, dup := byLSN[lsn]; dup {
							t.Errorf("LSN %d handed out twice: %q and %q", lsn, prev, p)
						}
						byLSN[lsn] = p
						mu.Unlock()
					}
				}()
			}
			wg.Wait()
			if t.Failed() {
				return
			}

			const total = writers * each
			if len(byLSN) != total || l.LastLSN() != total {
				t.Fatalf("%d distinct LSNs, LastLSN %d, want %d", len(byLSN), l.LastLSN(), total)
			}
			verify := func(l *Log) {
				t.Helper()
				next := uint64(1)
				err := l.Replay(1, func(lsn uint64, p []byte) error {
					if lsn != next {
						t.Fatalf("replay LSN %d, want %d", lsn, next)
					}
					if byLSN[lsn] != string(p) {
						t.Fatalf("LSN %d replayed %q, appended %q", lsn, p, byLSN[lsn])
					}
					next++
					return nil
				})
				if err != nil || next != total+1 {
					t.Fatalf("replayed through %d, err %v", next-1, err)
				}
			}
			verify(l)
			if opts.Sync == SyncAlways {
				// Not asserted as a ratio: how many appends share an fsync
				// depends on scheduling. That it is fewer than one each is
				// the property.
				if l.fsyncs >= total {
					t.Errorf("%d fsyncs for %d appends: no group commit", l.fsyncs, total)
				}
				t.Logf("%d appends, %d fsyncs", total, l.fsyncs)
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			verify(mustOpen(t, dir, opts))
		})
	}
}

// Replay promises everything appended before it was called, and must stay
// consistent while writers keep extending and rolling the log under it.
func TestReplayDuringAppends(t *testing.T) {
	dir := t.TempDir()
	l := mustOpen(t, dir, smallSegments)
	appendN(t, l, 1, 50)

	// A fixed amount of writing, so the test costs the same on a slow disk
	// or under the race detector; replays run for as long as it lasts.
	writing := make(chan struct{})
	go func() {
		defer close(writing)
		for i := 51; i <= 1500; i++ {
			if _, err := l.Append(payloadFor(i)); err != nil {
				t.Errorf("Append: %v", err)
				return
			}
		}
	}()

	for round, done := 0, false; !done; round++ {
		select {
		case <-writing:
			done = true // one more pass, over the finished log
		default:
		}
		floor := l.LastLSN()
		next := uint64(1)
		err := l.Replay(1, func(lsn uint64, p []byte) error {
			if lsn != next || !bytes.Equal(p, payloadFor(int(lsn))) {
				t.Fatalf("round %d: LSN %d (want %d) payload %q", round, lsn, next, p)
			}
			next++
			return nil
		})
		if err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		if next-1 < floor {
			t.Fatalf("round %d: replay ended at %d, but %d was appended before it started", round, next-1, floor)
		}
	}
	if l.LastLSN() != 1500 {
		t.Fatalf("LastLSN = %d, want 1500", l.LastLSN())
	}
}

func TestSyncIntervalSyncsInBackground(t *testing.T) {
	l := mustOpen(t, t.TempDir(), Options{Sync: SyncInterval, Interval: time.Millisecond})
	appendN(t, l, 1, 10)
	deadline := time.Now().Add(5 * time.Second)
	for {
		l.syncMu.Lock()
		durable := l.durable
		l.syncMu.Unlock()
		if durable == 10 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("background sync never ran: durable = %d", durable)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestSync(t *testing.T) {
	l := mustOpen(t, t.TempDir(), Options{Sync: SyncNever})
	appendN(t, l, 1, 5)
	if l.durable != 0 {
		t.Fatalf("SyncNever fsynced on append: durable = %d", l.durable)
	}
	if err := l.Sync(); err != nil {
		t.Fatal(err)
	}
	if l.durable != 5 {
		t.Fatalf("durable = %d after Sync", l.durable)
	}
}

func TestClose(t *testing.T) {
	for _, policy := range []SyncPolicy{SyncAlways, SyncInterval, SyncNever} {
		dir := t.TempDir()
		l, err := Open(dir, Options{Sync: policy})
		if err != nil {
			t.Fatal(err)
		}
		appendN(t, l, 1, 3)
		for i := 0; i < 3; i++ {
			if err := l.Close(); err != nil {
				t.Fatalf("Close #%d: %v", i+1, err)
			}
		}
		if policy == SyncInterval {
			select {
			case <-l.done:
			default:
				t.Fatal("sync goroutine still running after Close")
			}
		}
		if lsn, err := l.Append([]byte("late")); !errors.Is(err, ErrClosed) || lsn != 0 {
			t.Fatalf("Append after Close: lsn %d, err %v", lsn, err)
		}
		if err := l.Sync(); !errors.Is(err, ErrClosed) {
			t.Fatalf("Sync after Close: %v", err)
		}
		if err := l.Replay(1, func(uint64, []byte) error { return nil }); !errors.Is(err, ErrClosed) {
			t.Fatalf("Replay after Close: %v", err)
		}
		if err := l.TruncateBefore(2); !errors.Is(err, ErrClosed) {
			t.Fatalf("TruncateBefore after Close: %v", err)
		}
		if l.LastLSN() != 3 {
			t.Fatalf("LastLSN after Close = %d", l.LastLSN())
		}

		l = mustOpen(t, dir, Options{Sync: policy})
		checkReplay(t, l, 1, 3)
	}
}

// Close racing with appenders: every Append either fails with ErrClosed or
// returns an LSN that is in the log afterwards.
func TestCloseDuringAppends(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir, Options{Sync: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var acked uint64
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				lsn, err := l.Append([]byte("x"))
				if err != nil {
					if !errors.Is(err, ErrClosed) {
						t.Errorf("Append: %v", err)
					}
					return
				}
				mu.Lock()
				acked = max(acked, lsn)
				mu.Unlock()
			}
		}()
	}
	time.Sleep(20 * time.Millisecond)
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()

	l = mustOpen(t, dir, Options{})
	if l.LastLSN() < acked {
		t.Fatalf("acknowledged LSN %d, log ends at %d", acked, l.LastLSN())
	}
}

func TestIgnoresForeignFiles(t *testing.T) {
	dir := buildLog(t, 20)
	for _, name := range []string{"snapshot.bin", "00000000000000000001.wal.tmp", "1.wal"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("not a segment"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	l := mustOpen(t, dir, smallSegments)
	checkReplay(t, l, 1, 20)
}
