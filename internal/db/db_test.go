package db

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/rishiraj38/quiver/internal/wal"
)

const testDim = 16

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func mustOpen(t *testing.T, dir string) *DB {
	t.Helper()
	// SyncNever keeps the suite fast; recovery from real kills is covered by
	// the process-level test in /test.
	d, err := Open(dir, Options{WAL: wal.Options{Sync: wal.SyncNever}})
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustCreate(t *testing.T, d *DB, name string, cfg CollectionConfig) *Collection {
	t.Helper()
	c, err := d.Create(name, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func makePoints(r *rand.Rand, from, n int) []Point {
	ps := make([]Point, n)
	for i := range ps {
		id := from + i
		ps[i] = Point{
			ID:      uint64(id),
			Vector:  randVec(r, testDim),
			Payload: json.RawMessage(fmt.Sprintf(`{"n":%d,"parity":%q}`, id, []string{"even", "odd"}[id%2])),
		}
	}
	return ps
}

// topIDs runs the same query on a collection and returns the ids, so two
// states of a collection can be compared.
func topIDs(t *testing.T, c *Collection, q []float32, k int) []uint64 {
	t.Helper()
	hits, err := c.Search(SearchRequest{Vector: q, K: k, Ef: 200})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, len(hits))
	for i, h := range hits {
		ids[i] = h.ID
	}
	return ids
}

func TestUpsertGetSearchDelete(t *testing.T) {
	d := mustOpen(t, t.TempDir())
	defer d.Close()
	c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim})
	r := rand.New(rand.NewPCG(1, 2))
	points := makePoints(r, 0, 500)
	if err := c.Upsert(points); err != nil {
		t.Fatal(err)
	}
	if got := c.Stats().Count; got != 500 {
		t.Fatalf("count = %d, want 500", got)
	}

	// A stored vector is its own nearest neighbour.
	hits, err := c.Search(SearchRequest{Vector: points[42].Vector, K: 3, WithPayload: true, WithVector: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 || hits[0].ID != 42 || hits[0].Distance > 1e-6 {
		t.Fatalf("self search returned %+v", hits)
	}
	if hits[0].Payload["n"] != float64(42) || len(hits[0].Vector) != testDim {
		t.Fatalf("payload or vector missing: %+v", hits[0])
	}

	// Upsert replaces vector and payload without changing the count.
	repl := Point{ID: 42, Vector: randVec(r, testDim), Payload: json.RawMessage(`{"n":-1}`)}
	if err := c.Upsert([]Point{repl}); err != nil {
		t.Fatal(err)
	}
	got, err := c.Get(42)
	if err != nil {
		t.Fatal(err)
	}
	if got.Payload["n"] != float64(-1) || got.Vector[0] != repl.Vector[0] {
		t.Fatalf("upsert did not replace: %+v", got)
	}
	if got := c.Stats().Count; got != 500 {
		t.Fatalf("count after replace = %d, want 500", got)
	}

	n, err := c.Delete([]uint64{42, 43, 99999})
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("deleted %d, want 2", n)
	}
	if _, err := c.Get(42); !errors.Is(err, ErrNotFound) {
		t.Fatalf("get after delete: %v", err)
	}
	for _, id := range topIDs(t, c, repl.Vector, 20) {
		if id == 42 || id == 43 {
			t.Fatalf("deleted id %d returned by search", id)
		}
	}
}

func TestFilteredSearch(t *testing.T) {
	d := mustOpen(t, t.TempDir())
	defer d.Close()
	c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim})
	r := rand.New(rand.NewPCG(3, 4))
	if err := c.Upsert(makePoints(r, 0, 2000)); err != nil {
		t.Fatal(err)
	}
	f := &Filter{Must: []Condition{
		{Key: "parity", Op: "eq", Value: "odd"},
		{Key: "n", Op: "lt", Value: float64(200)},
	}}
	hits, err := c.Search(SearchRequest{Vector: randVec(r, testDim), K: 25, Filter: f, WithPayload: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 25 {
		t.Fatalf("got %d hits, want 25 (100 points match)", len(hits))
	}
	for _, h := range hits {
		if h.ID%2 != 1 || h.ID >= 200 {
			t.Fatalf("hit %d does not satisfy the filter", h.ID)
		}
	}
	_, err = c.Search(SearchRequest{Vector: randVec(r, testDim), Filter: &Filter{Must: []Condition{{Key: "n", Op: "nope"}}}})
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("bad filter: got %v, want ErrInvalid", err)
	}
}

func TestValidation(t *testing.T) {
	d := mustOpen(t, t.TempDir())
	defer d.Close()
	c := mustCreate(t, d, "cos", CollectionConfig{Dim: 4, Metric: "cosine"})
	nan := float32(0) / float32(1)
	nan = nan / nan
	bad := [][]Point{
		{{ID: 1, Vector: []float32{1, 2, 3}}},
		{{ID: 1, Vector: []float32{1, 2, 3, nan}}},
		{{ID: 1, Vector: []float32{0, 0, 0, 0}}},
		{{ID: 1, Vector: []float32{1, 0, 0, 0}}, {ID: 1, Vector: []float32{0, 1, 0, 0}}},
		{{ID: 1, Vector: []float32{1, 0, 0, 0}, Payload: json.RawMessage(`[1,2]`)}},
	}
	for i, ps := range bad {
		if err := c.Upsert(ps); !errors.Is(err, ErrInvalid) {
			t.Errorf("case %d: got %v, want ErrInvalid", i, err)
		}
	}
	// Nothing invalid may have reached the log.
	if st := c.Stats(); st.Count != 0 || st.LastLSN != 0 {
		t.Fatalf("rejected writes left state behind: %+v", st)
	}
	if _, err := d.Create("cos", CollectionConfig{Dim: 4}); !errors.Is(err, ErrExists) {
		t.Errorf("duplicate create: %v", err)
	}
	for _, name := range []string{"", "../x", "a b", "a/b"} {
		if _, err := d.Create(name, CollectionConfig{Dim: 4}); !errors.Is(err, ErrInvalid) {
			t.Errorf("create %q: got %v, want ErrInvalid", name, err)
		}
	}
	if _, err := d.Create("x", CollectionConfig{Dim: 0}); !errors.Is(err, ErrInvalid) {
		t.Errorf("dim 0: %v", err)
	}
	if _, err := d.Create("x", CollectionConfig{Dim: 4, Metric: "hamming"}); !errors.Is(err, ErrInvalid) {
		t.Errorf("bad metric: %v", err)
	}
}

// The three recovery shapes: log only, snapshot only, snapshot plus a log
// tail written after it. Each must come back answering queries identically.
func TestRecovery(t *testing.T) {
	for _, mode := range []string{"wal-only", "snapshot-only", "snapshot-plus-tail"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			d := mustOpen(t, dir)
			c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim, Metric: "cosine"})
			r := rand.New(rand.NewPCG(5, 6))
			if err := c.Upsert(makePoints(r, 0, 1500)); err != nil {
				t.Fatal(err)
			}
			if _, err := c.Delete([]uint64{10, 11, 12}); err != nil {
				t.Fatal(err)
			}
			if mode != "wal-only" {
				if err := c.Snapshot(); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "snapshot-plus-tail" {
				if err := c.Upsert(makePoints(r, 1500, 300)); err != nil {
					t.Fatal(err)
				}
				// Re-add a deleted id, overwrite a live one, delete another.
				if err := c.Upsert(makePoints(r, 10, 1)); err != nil {
					t.Fatal(err)
				}
				if err := c.Upsert([]Point{{ID: 7, Vector: randVec(r, testDim), Payload: json.RawMessage(`{"n":7000}`)}}); err != nil {
					t.Fatal(err)
				}
				if _, err := c.Delete([]uint64{20}); err != nil {
					t.Fatal(err)
				}
			}
			want := c.Stats()
			queries := make([][]float32, 20)
			before := make([][]uint64, len(queries))
			for i := range queries {
				queries[i] = randVec(r, testDim)
				before[i] = topIDs(t, c, queries[i], 10)
			}
			p7, _ := c.Get(7)
			if err := d.Close(); err != nil {
				t.Fatal(err)
			}

			d2 := mustOpen(t, dir)
			defer d2.Close()
			c2, err := d2.Get("items")
			if err != nil {
				t.Fatal(err)
			}
			got := c2.Stats()
			if got.Count != want.Count || got.LastLSN != want.LastLSN || got.SnapshotLSN != want.SnapshotLSN {
				t.Fatalf("stats after reopen = %+v, want %+v", got, want)
			}
			if got.Config.Metric != "cosine" {
				t.Fatalf("config lost: %+v", got.Config)
			}
			// A snapshot restores the exact graph, so results must be
			// identical. A replayed log rebuilds the graph with parallel
			// inserts, so it is only held to the same recall.
			overlap, total := 0, 0
			for i, q := range queries {
				after := topIDs(t, c2, q, 10)
				total += len(before[i])
				for _, id := range after {
					for _, b := range before[i] {
						if id == b {
							overlap++
						}
					}
				}
				if mode == "snapshot-only" && fmt.Sprint(after) != fmt.Sprint(before[i]) {
					t.Fatalf("query %d: %v after reopen, %v before", i, after, before[i])
				}
			}
			if frac := float64(overlap) / float64(total); frac < 0.95 {
				t.Fatalf("only %.2f of results survived recovery", frac)
			}
			p7b, err := c2.Get(7)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprint(p7b.Payload) != fmt.Sprint(p7.Payload) {
				t.Fatalf("payload of 7: %v after reopen, %v before", p7b.Payload, p7.Payload)
			}
			for _, id := range []uint64{11, 12} {
				if _, err := c2.Get(id); !errors.Is(err, ErrNotFound) {
					t.Fatalf("deleted id %d came back", id)
				}
			}
			// The reopened collection keeps accepting writes.
			if err := c2.Upsert(makePoints(r, 5000, 10)); err != nil {
				t.Fatal(err)
			}
			if c2.Stats().LastLSN != want.LastLSN+1 {
				t.Fatalf("lsn did not continue: %d after %d", c2.Stats().LastLSN, want.LastLSN)
			}
		})
	}
}

func TestSnapshotTrimsLog(t *testing.T) {
	dir := t.TempDir()
	d, err := Open(dir, Options{WAL: wal.Options{Sync: wal.SyncNever, SegmentSize: 16 << 10}})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim})
	r := rand.New(rand.NewPCG(7, 8))
	for i := range 40 {
		if err := c.Upsert(makePoints(r, i*50, 50)); err != nil {
			t.Fatal(err)
		}
	}
	walDir := filepath.Join(dir, "items", "wal")
	before, _ := os.ReadDir(walDir)
	if len(before) < 3 {
		t.Fatalf("expected several segments, got %d", len(before))
	}
	if err := c.Snapshot(); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadDir(walDir)
	if len(after) >= len(before) {
		t.Fatalf("snapshot left %d segments, had %d", len(after), len(before))
	}
	if st := c.Stats(); st.SnapshotLSN != st.LastLSN {
		t.Fatalf("snapshot lsn %d, last lsn %d", st.SnapshotLSN, st.LastLSN)
	}
}

func TestCorruptSnapshotRefusesToOpen(t *testing.T) {
	dir := t.TempDir()
	d := mustOpen(t, dir)
	c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim})
	if err := c.Upsert(makePoints(rand.New(rand.NewPCG(9, 9)), 0, 200)); err != nil {
		t.Fatal(err)
	}
	if err := c.Snapshot(); err != nil {
		t.Fatal(err)
	}
	d.Close()
	path := filepath.Join(dir, "items", snapshotName)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, off := range []int{30, len(raw) / 2, len(raw) - 10} {
		bad := append([]byte(nil), raw...)
		bad[off] ^= 0xFF
		if err := os.WriteFile(path, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if d2, err := Open(dir, Options{}); err == nil {
			d2.Close()
			t.Fatalf("opened with snapshot byte %d flipped", off)
		}
	}
}

func TestDrop(t *testing.T) {
	dir := t.TempDir()
	d := mustOpen(t, dir)
	mustCreate(t, d, "gone", CollectionConfig{Dim: testDim})
	mustCreate(t, d, "kept", CollectionConfig{Dim: testDim})
	if err := d.Drop("gone"); err != nil {
		t.Fatal(err)
	}
	if err := d.Drop("gone"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second drop: %v", err)
	}
	d.Close()
	d2 := mustOpen(t, dir)
	defer d2.Close()
	if list := d2.List(); len(list) != 1 || list[0].Name != "kept" {
		t.Fatalf("after reopen: %+v", list)
	}
}

// Writers on overlapping ids, searches and snapshots all at once. Run with
// -race. Afterwards every id must hold the value of its last write, both in
// memory and after replaying the log.
func TestConcurrentWritesSearchSnapshot(t *testing.T) {
	dir := t.TempDir()
	d := mustOpen(t, dir)
	c := mustCreate(t, d, "items", CollectionConfig{Dim: testDim})
	const (
		writers = 8
		rounds  = 60
		ids     = 40
	)
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := rand.New(rand.NewPCG(uint64(w), 11))
			for i := range rounds {
				id := uint64(r.IntN(ids))
				switch {
				case i%7 == 6:
					if _, err := c.Delete([]uint64{id}); err != nil {
						t.Error(err)
					}
				case i%5 == 4:
					batch := makePoints(r, int(id), 3)
					if err := c.Upsert(batch); err != nil {
						t.Error(err)
					}
				default:
					p := Point{ID: id, Vector: randVec(r, testDim), Payload: json.RawMessage(fmt.Sprintf(`{"w":%d,"i":%d}`, w, i))}
					if err := c.Upsert([]Point{p}); err != nil {
						t.Error(err)
					}
				}
			}
		}()
	}
	var bg sync.WaitGroup
	bg.Add(2)
	go func() {
		defer bg.Done()
		r := rand.New(rand.NewPCG(99, 99))
		for {
			select {
			case <-stop:
				return
			default:
				if _, err := c.Search(SearchRequest{Vector: randVec(r, testDim), K: 5, WithPayload: true}); err != nil {
					t.Error(err)
				}
			}
		}
	}()
	go func() {
		defer bg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				if err := c.Snapshot(); err != nil {
					t.Error(err)
				}
			}
		}
	}()
	wg.Wait()
	close(stop)
	bg.Wait()

	type state struct {
		payload string
		v0      float32
	}
	dump := func(c *Collection) map[uint64]state {
		m := make(map[uint64]state)
		for id := uint64(0); id < ids+3; id++ {
			if p, err := c.Get(id); err == nil {
				m[id] = state{fmt.Sprint(p.Payload), p.Vector[0]}
			}
		}
		return m
	}
	want := dump(c)
	if len(want) != c.Stats().Count {
		t.Fatalf("index holds %d points but %d ids are readable", c.Stats().Count, len(want))
	}
	d.Close()

	d2 := mustOpen(t, dir)
	defer d2.Close()
	c2, _ := d2.Get("items")
	got := dump(c2)
	if len(got) != len(want) {
		t.Fatalf("%d points after recovery, %d before", len(got), len(want))
	}
	for id, w := range want {
		if got[id] != w {
			t.Fatalf("id %d: %+v after recovery, %+v before", id, got[id], w)
		}
	}
}

func TestDataDirIsLocked(t *testing.T) {
	dir := t.TempDir()
	d := mustOpen(t, dir)
	if _, err := Open(dir, Options{}); !errors.Is(err, ErrLocked) {
		t.Fatalf("second open: got %v, want ErrLocked", err)
	}
	d.Close()
	d2 := mustOpen(t, dir)
	d2.Close()
}
