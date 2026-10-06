package hnsw

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rishiraj38/quiver/internal/vec"
)

func randVecs(seed int64, n, dim int) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	flat := make([]float32, n*dim)
	for i := range flat {
		flat[i] = rng.Float32()*2 - 1
	}
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*dim : (i+1)*dim]
	}
	return out
}

// buildParallel inserts vecs[i] under id i using every core.
func buildParallel(tb testing.TB, ix *Index, vecs [][]float32) {
	tb.Helper()
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(vecs) {
					return
				}
				if err := ix.Add(uint64(i), vecs[i]); err != nil {
					tb.Errorf("Add(%d): %v", i, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// bruteForce is the ground truth: an exact scan under the index's metric.
func bruteForce(metric vec.Metric, vecs [][]float32, q []float32, k int, keep func(id uint64) bool) []Result {
	dist := metric.Func()
	if metric == vec.Cosine {
		q = slices.Clone(q)
		vec.Normalize(q)
	}
	var all []Result
	for i, v := range vecs {
		if keep != nil && !keep(uint64(i)) {
			continue
		}
		if metric == vec.Cosine {
			v = slices.Clone(v)
			vec.Normalize(v)
		}
		all = append(all, Result{ID: uint64(i), Dist: dist(q, v)})
	}
	slices.SortFunc(all, func(a, b Result) int {
		if a.Dist != b.Dist {
			if a.Dist < b.Dist {
				return -1
			}
			return 1
		}
		return 0
	})
	return all[:min(k, len(all))]
}

func recall(got, want []Result) float64 {
	truth := make(map[uint64]bool, len(want))
	for _, r := range want {
		truth[r.ID] = true
	}
	hit := 0
	for _, r := range got {
		if truth[r.ID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

func checkSorted(t *testing.T, res []Result) {
	t.Helper()
	for i := 1; i < len(res); i++ {
		if res[i].Dist < res[i-1].Dist {
			t.Fatalf("results not sorted: %v before %v", res[i-1], res[i])
		}
	}
}

// Uniform random vectors are the worst case for any graph index: with no
// structure to exploit, the intrinsic dimension equals the nominal one. The
// floors below sit just under what this implementation and FAISS's
// IndexHNSWFlat both reach on this distribution with the same parameters
// (about 0.90 at ef=100 and 0.975 at ef=200).
func TestRecall(t *testing.T) {
	n, queries := 20000, 200
	if testing.Short() {
		n, queries = 4000, 50
	}
	const dim, k = 64, 10
	floors := []struct {
		ef  int
		min float64
	}{{100, 0.88}, {200, 0.95}}
	for _, metric := range []vec.Metric{vec.L2, vec.Cosine} {
		t.Run(metric.String(), func(t *testing.T) {
			vecs := randVecs(1, n, dim)
			ix := New(dim, Config{Metric: metric})
			buildParallel(t, ix, vecs)
			if ix.Len() != n {
				t.Fatalf("Len = %d, want %d", ix.Len(), n)
			}
			qs := randVecs(2, queries, dim)
			truth := make([][]Result, queries)
			for i, q := range qs {
				truth[i] = bruteForce(metric, vecs, q, k, nil)
			}
			for _, f := range floors {
				var sum float64
				for i, q := range qs {
					got := ix.Search(q, k, f.ef, nil)
					if len(got) != k {
						t.Fatalf("got %d results, want %d", len(got), k)
					}
					checkSorted(t, got)
					sum += recall(got, truth[i])
				}
				r := sum / float64(queries)
				t.Logf("%s recall@%d = %.4f (n=%d, ef=%d)", metric, k, r, n, f.ef)
				if r < f.min {
					t.Fatalf("recall@%d = %.4f at ef=%d, want >= %.2f", k, r, f.ef, f.min)
				}
			}
		})
	}
}

// Clustered data is where closest-M selection falls apart and the heuristic
// earns its keep, so it gets its own recall floor.
func TestRecallClustered(t *testing.T) {
	const dim, k, clusters = 32, 10, 50
	n := 10000
	if testing.Short() {
		n = 3000
	}
	rng := rand.New(rand.NewSource(7))
	centres := randVecs(8, clusters, dim)
	vecs := make([][]float32, n)
	for i := range vecs {
		c := centres[rng.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())*0.02
		}
		vecs[i] = v
	}
	ix := New(dim, Config{})
	buildParallel(t, ix, vecs)

	var sum float64
	const queries = 100
	for i := 0; i < queries; i++ {
		q := vecs[rng.Intn(n)]
		sum += recall(ix.Search(q, k, 100, nil), bruteForce(vec.L2, vecs, q, k, nil))
	}
	if r := sum / queries; r < 0.95 {
		t.Fatalf("clustered recall@%d = %.4f, want >= 0.95", k, r)
	}
}

func TestEmptyAndSmall(t *testing.T) {
	ix := New(4, Config{})
	if got := ix.Search([]float32{0, 0, 0, 0}, 5, 10, nil); len(got) != 0 {
		t.Fatalf("empty index returned %v", got)
	}
	if ix.Len() != 0 || ix.Dim() != 4 {
		t.Fatalf("Len=%d Dim=%d", ix.Len(), ix.Dim())
	}
	if ix.Delete(1) {
		t.Fatal("Delete on empty index reported true")
	}
	if _, ok := ix.Vector(1); ok {
		t.Fatal("Vector on empty index reported ok")
	}

	for i := 0; i < 3; i++ {
		if err := ix.Add(uint64(10+i), []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	got := ix.Search([]float32{0.1, 0, 0, 0}, 10, 1, nil)
	if len(got) != 3 {
		t.Fatalf("k > Len: got %d results, want 3", len(got))
	}
	for i, r := range got {
		if r.ID != uint64(10+i) {
			t.Fatalf("result %d = %v, want id %d", i, r, 10+i)
		}
	}
	if d := got[0].Dist; math.Abs(float64(d)-0.01) > 1e-6 {
		t.Fatalf("nearest distance = %v, want 0.01", d)
	}

	if got := ix.Search([]float32{0, 0, 0, 0}, 0, 10, nil); got != nil {
		t.Fatalf("k=0 returned %v", got)
	}
	if got := ix.Search([]float32{0, 0}, 1, 10, nil); got != nil {
		t.Fatalf("wrong-dimension query returned %v", got)
	}
}

func TestConfigDefaults(t *testing.T) {
	cfg := New(8, Config{}).Config()
	if cfg.M != 16 || cfg.EfConstruction != 200 || cfg.Metric != vec.L2 || cfg.Seed == 0 {
		t.Fatalf("defaults not applied: %+v", cfg)
	}
	cfg = New(8, Config{M: 8, EfConstruction: 50, Metric: vec.Cosine, Seed: 9}).Config()
	if cfg.M != 8 || cfg.EfConstruction != 50 || cfg.Metric != vec.Cosine || cfg.Seed != 9 {
		t.Fatalf("explicit config changed: %+v", cfg)
	}
}

func TestErrors(t *testing.T) {
	ix := New(3, Config{})
	if err := ix.Add(1, []float32{1, 2}); !errors.Is(err, ErrDimension) {
		t.Fatalf("short vector: err = %v, want ErrDimension", err)
	}
	if err := ix.Add(1, []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(1, []float32{4, 5, 6}); !errors.Is(err, ErrExists) {
		t.Fatalf("duplicate id: err = %v, want ErrExists", err)
	}
	if ix.Len() != 1 {
		t.Fatalf("Len = %d after rejected adds, want 1", ix.Len())
	}
	v, ok := ix.Vector(1)
	if !ok || !slices.Equal(v, []float32{1, 2, 3}) {
		t.Fatalf("Vector(1) = %v, %v", v, ok)
	}
}

func TestAddCopiesVector(t *testing.T) {
	ix := New(2, Config{})
	v := []float32{1, 2}
	if err := ix.Add(1, v); err != nil {
		t.Fatal(err)
	}
	v[0] = 99
	got, _ := ix.Vector(1)
	if got[0] != 1 {
		t.Fatal("index aliases the caller's slice")
	}
	got[1] = 99
	if again, _ := ix.Vector(1); again[1] != 2 {
		t.Fatal("Vector returns the index's own storage")
	}
}

func TestCosineNormalises(t *testing.T) {
	ix := New(2, Config{Metric: vec.Cosine})
	if err := ix.Add(1, []float32{3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(2, []float32{-5, 0}); err != nil {
		t.Fatal(err)
	}
	v, _ := ix.Vector(1)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("stored vector = %v, want unit length", v)
	}
	q := []float32{30, 40}
	got := ix.Search(q, 2, 10, nil)
	if got[0].ID != 1 || math.Abs(float64(got[0].Dist)) > 1e-6 {
		t.Fatalf("scaled query: nearest = %v, want id 1 at distance 0", got[0])
	}
	if q[0] != 30 {
		t.Fatal("Search modified the caller's query")
	}
}

func TestDeleteAndReAdd(t *testing.T) {
	const n, dim = 2000, 16
	vecs := randVecs(3, n, dim)
	ix := New(dim, Config{})
	buildParallel(t, ix, vecs)

	deleted := func(id uint64) bool { return id%3 == 0 }
	for i := 0; i < n; i++ {
		if deleted(uint64(i)) && !ix.Delete(uint64(i)) {
			t.Fatalf("Delete(%d) = false", i)
		}
	}
	if ix.Delete(0) {
		t.Fatal("second Delete reported true")
	}
	want := n - (n+2)/3
	if ix.Len() != want {
		t.Fatalf("Len = %d, want %d", ix.Len(), want)
	}
	if _, ok := ix.Vector(0); ok {
		t.Fatal("Vector returned a deleted id")
	}

	var sum float64
	queries := randVecs(4, 50, dim)
	for _, q := range queries {
		got := ix.Search(q, 10, 100, nil)
		if len(got) != 10 {
			t.Fatalf("got %d results after deletes, want 10", len(got))
		}
		for _, r := range got {
			if deleted(r.ID) {
				t.Fatalf("deleted id %d returned", r.ID)
			}
		}
		keep := func(id uint64) bool { return !deleted(id) }
		sum += recall(got, bruteForce(vec.L2, vecs, q, 10, keep))
	}
	if r := sum / float64(len(queries)); r < 0.95 {
		t.Fatalf("recall after deletes = %.4f, want >= 0.95", r)
	}

	// Re-add id 0 somewhere else entirely; the old vector must be gone.
	moved := make([]float32, dim)
	for i := range moved {
		moved[i] = 50
	}
	if err := ix.Add(0, moved); err != nil {
		t.Fatalf("re-add: %v", err)
	}
	if err := ix.Add(0, moved); !errors.Is(err, ErrExists) {
		t.Fatalf("re-add twice: err = %v, want ErrExists", err)
	}
	if ix.Len() != want+1 {
		t.Fatalf("Len = %d after re-add, want %d", ix.Len(), want+1)
	}
	got := ix.Search(moved, 1, 50, nil)
	if len(got) != 1 || got[0].ID != 0 || got[0].Dist != 0 {
		t.Fatalf("search at re-added vector = %v", got)
	}
	if v, ok := ix.Vector(0); !ok || !slices.Equal(v, moved) {
		t.Fatal("Vector does not return the re-added vector")
	}
	for _, r := range ix.Search(vecs[0], 10, 100, nil) {
		if r.ID == 0 {
			t.Fatal("old copy of a re-added id is still returned")
		}
	}
}

func TestDeleteEverything(t *testing.T) {
	ix := New(4, Config{})
	vecs := randVecs(5, 200, 4)
	for i, v := range vecs {
		if err := ix.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	for i := range vecs {
		ix.Delete(uint64(i))
	}
	if got := ix.Search(vecs[0], 5, 50, nil); len(got) != 0 {
		t.Fatalf("all deleted, got %v", got)
	}
	if err := ix.Add(1000, vecs[0]); err != nil {
		t.Fatal(err)
	}
	if got := ix.Search(vecs[0], 5, 50, nil); len(got) != 1 || got[0].ID != 1000 {
		t.Fatalf("after re-populating: %v", got)
	}
}

func TestFilter(t *testing.T) {
	const n, dim, k = 10000, 32, 10
	vecs := randVecs(6, n, dim)
	ix := New(dim, Config{})
	buildParallel(t, ix, vecs)

	allow := func(id uint64) bool { return id%20 == 0 } // 5% pass
	var sum float64
	queries := randVecs(7, 50, dim)
	for _, q := range queries {
		got := ix.Search(q, k, 50, allow)
		if len(got) != k {
			t.Fatalf("5%% filter: got %d results, want %d", len(got), k)
		}
		checkSorted(t, got)
		for _, r := range got {
			if !allow(r.ID) {
				t.Fatalf("id %d does not satisfy the filter", r.ID)
			}
		}
		sum += recall(got, bruteForce(vec.L2, vecs, q, k, allow))
	}
	if r := sum / float64(len(queries)); r < 0.95 {
		t.Fatalf("filtered recall = %.4f, want >= 0.95", r)
	}

	if got := ix.Search(queries[0], k, 50, func(uint64) bool { return false }); len(got) != 0 {
		t.Fatalf("reject-all filter returned %v", got)
	}
	only := func(id uint64) bool { return id == 4242 }
	if got := ix.Search(queries[0], k, 50, only); len(got) != 1 || got[0].ID != 4242 {
		t.Fatalf("single-id filter returned %v", got)
	}
}

// A filter may call back into the index; that must not deadlock.
func TestFilterReentrant(t *testing.T) {
	ix := New(4, Config{})
	vecs := randVecs(8, 100, 4)
	for i, v := range vecs {
		if err := ix.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	got := ix.Search(vecs[0], 5, 20, func(id uint64) bool {
		_, ok := ix.Vector(id)
		return ok && ix.Len() > 0
	})
	if len(got) != 5 {
		t.Fatalf("got %d results, want 5", len(got))
	}
}

func TestConcurrentAddSearchDelete(t *testing.T) {
	const dim, writers, perWriter = 24, 8, 600
	const total = writers * perWriter
	vecs := randVecs(9, total, dim)
	ix := New(dim, Config{M: 8, EfConstruction: 64})

	var writing, searching sync.WaitGroup
	stop := make(chan struct{})
	for r := 0; r < 4; r++ {
		searching.Add(1)
		go func(r int) {
			defer searching.Done()
			rng := rand.New(rand.NewSource(int64(100 + r)))
			var filter func(uint64) bool
			if r%2 == 1 {
				filter = func(id uint64) bool { return id%2 == 0 }
			}
			for {
				select {
				case <-stop:
					return
				default:
				}
				res := ix.Search(vecs[rng.Intn(total)], 10, 40, filter)
				for i, x := range res {
					if x.ID >= total || (filter != nil && !filter(x.ID)) {
						t.Errorf("bad result %v", x)
						return
					}
					if i > 0 && x.Dist < res[i-1].Dist {
						t.Errorf("unsorted results")
						return
					}
				}
				ix.Len()
				ix.Vector(uint64(rng.Intn(total)))
			}
		}(r)
	}
	for w := 0; w < writers; w++ {
		writing.Add(1)
		go func(w int) {
			defer writing.Done()
			for i := w * perWriter; i < (w+1)*perWriter; i++ {
				if err := ix.Add(uint64(i), vecs[i]); err != nil {
					t.Errorf("Add(%d): %v", i, err)
					return
				}
				// Delete every tenth id and put every twentieth back, so
				// tombstones and re-adds happen under load too.
				if i%10 == 0 {
					if !ix.Delete(uint64(i)) {
						t.Errorf("Delete(%d) = false", i)
					}
					if i%20 == 0 {
						if err := ix.Add(uint64(i), vecs[i]); err != nil {
							t.Errorf("re-Add(%d): %v", i, err)
						}
					}
				}
			}
		}(w)
	}
	// Snapshot while everything is moving: must be loadable and coherent.
	var snap bytes.Buffer
	if _, err := ix.WriteTo(&snap); err != nil {
		t.Fatal(err)
	}
	writing.Wait()
	close(stop)
	searching.Wait()

	if _, err := Load(&snap); err != nil {
		t.Fatalf("snapshot taken under load does not load: %v", err)
	}
	want := total - total/10 + total/20
	if ix.Len() != want {
		t.Fatalf("Len = %d, want %d", ix.Len(), want)
	}
	checkGraph(t, ix)

	// Every live vector must still be findable as its own nearest neighbour.
	miss := 0
	for i := 0; i < total; i++ {
		live := i%10 != 0 || i%20 == 0
		res := ix.Search(vecs[i], 1, 64, nil)
		if live && (len(res) == 0 || res[0].ID != uint64(i)) {
			miss++
		}
		if !live && len(res) > 0 && res[0].ID == uint64(i) {
			t.Fatalf("deleted id %d returned", i)
		}
	}
	if miss > want/100 {
		t.Fatalf("%d of %d live vectors are not their own nearest neighbour", miss, want)
	}
}

// Concurrent Adds of one id: exactly one may win.
func TestConcurrentDuplicateAdd(t *testing.T) {
	ix := New(4, Config{})
	var wins atomic.Int32
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch err := ix.Add(7, []float32{1, 2, 3, 4}); {
			case err == nil:
				wins.Add(1)
			case !errors.Is(err, ErrExists):
				t.Errorf("unexpected error %v", err)
			}
		}()
	}
	wg.Wait()
	if wins.Load() != 1 || ix.Len() != 1 {
		t.Fatalf("wins = %d, Len = %d", wins.Load(), ix.Len())
	}
}

// checkGraph asserts the structural invariants: degree bounds, no self
// loops, no duplicate edges, and every edge pointing at a node that exists
// on that layer.
func checkGraph(t *testing.T, ix *Index) {
	t.Helper()
	tab := *ix.table.Load()
	for slot := uint32(0); slot < ix.count; slot++ {
		c := tab[slot>>chunkShift]
		i := slot & chunkMask
		for layer := 0; layer <= int(c.nodes[i].level); layer++ {
			l := ix.links(c, i, layer)
			limit := ix.cfg.M
			if layer == 0 {
				limit = 2 * ix.cfg.M
			}
			if int(l[0]) > limit {
				t.Fatalf("slot %d layer %d has %d neighbours, limit %d", slot, layer, l[0], limit)
			}
			seen := map[uint32]bool{}
			for _, nb := range l[1 : 1+l[0]] {
				if nb == slot || nb >= ix.count || seen[nb] {
					t.Fatalf("slot %d layer %d: bad edge to %d", slot, layer, nb)
				}
				seen[nb] = true
				if int(tab[nb>>chunkShift].nodes[nb&chunkMask].level) < layer {
					t.Fatalf("slot %d layer %d: neighbour %d does not reach this layer", slot, layer, nb)
				}
			}
		}
	}
}

func TestLevelDistribution(t *testing.T) {
	ix := New(4, Config{M: 16})
	const n = 200000
	var hist [maxLevel + 1]int
	for slot := uint32(0); slot < n; slot++ {
		hist[ix.levelFor(slot)]++
	}
	// P(level >= l) = M^-l, so each layer should hold about 1/16 of the
	// one below it.
	for l := 1; l <= 2; l++ {
		above := 0
		for _, c := range hist[l:] {
			above += c
		}
		want := float64(n) / math.Pow(16, float64(l))
		if got := float64(above); got < want*0.8 || got > want*1.2 {
			t.Fatalf("%d nodes at level >= %d, want about %.0f", above, l, want)
		}
	}
}

func TestDeterministicBuild(t *testing.T) {
	vecs := randVecs(10, 3000, 16)
	build := func() []byte {
		ix := New(16, Config{Seed: 5})
		for i, v := range vecs {
			if err := ix.Add(uint64(i), v); err != nil {
				t.Fatal(err)
			}
		}
		var buf bytes.Buffer
		if _, err := ix.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	if !bytes.Equal(build(), build()) {
		t.Fatal("two single-threaded builds with the same seed differ")
	}
}

func TestVisitedEpochWrap(t *testing.T) {
	const n, dim = 500, 8
	vecs := randVecs(11, n, dim)
	ix := New(dim, Config{})
	for i, v := range vecs {
		if err := ix.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	// Force the wrap on the next search instead of running 65536 of them.
	s := ix.getScratch()
	s.begin(n)
	for i := range s.visited {
		s.visited[i] = 1
	}
	s.epoch = math.MaxUint16
	ix.pool.Put(s)
	for i := 0; i < 3; i++ {
		got := ix.Search(vecs[i], 1, 50, nil)
		if len(got) != 1 || got[0].ID != uint64(i) {
			t.Fatalf("search %d after epoch wrap = %v", i, got)
		}
	}
}

func buildSaved(t *testing.T, metric vec.Metric) (*Index, [][]float32, []byte) {
	t.Helper()
	const n, dim = 6000, 24 // spans two chunks
	vecs := randVecs(12, n, dim)
	ix := New(dim, Config{M: 12, EfConstruction: 100, Metric: metric, Seed: 77})
	buildParallel(t, ix, vecs)
	for i := 0; i < n; i += 7 {
		ix.Delete(uint64(i))
	}
	if err := ix.Add(0, vecs[1]); err != nil { // re-add after delete
		t.Fatal(err)
	}
	var buf bytes.Buffer
	written, err := ix.WriteTo(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if written != int64(buf.Len()) {
		t.Fatalf("WriteTo reported %d bytes, wrote %d", written, buf.Len())
	}
	return ix, vecs, buf.Bytes()
}

func TestSaveLoadRoundTrip(t *testing.T) {
	for _, metric := range []vec.Metric{vec.L2, vec.Cosine, vec.Dot} {
		t.Run(metric.String(), func(t *testing.T) {
			ix, vecs, data := buildSaved(t, metric)

			// Trailing bytes stand in for whatever follows the index in a
			// larger file; Load must leave them unread.
			r := bytes.NewReader(append(slices.Clone(data), "tail"...))
			got, err := Load(r)
			if err != nil {
				t.Fatal(err)
			}
			if r.Len() != 4 {
				t.Fatalf("Load left %d bytes unread, want 4", r.Len())
			}
			if got.Len() != ix.Len() || got.Dim() != ix.Dim() || got.Config() != ix.Config() {
				t.Fatalf("loaded Len=%d Dim=%d Config=%+v", got.Len(), got.Dim(), got.Config())
			}
			checkGraph(t, got)

			allow := func(id uint64) bool { return id%3 != 0 }
			for _, q := range randVecs(13, 200, ix.Dim()) {
				if a, b := ix.Search(q, 10, 80, nil), got.Search(q, 10, 80, nil); !slices.Equal(a, b) {
					t.Fatalf("results differ after load:\n%v\n%v", a, b)
				}
				if a, b := ix.Search(q, 10, 80, allow), got.Search(q, 10, 80, allow); !slices.Equal(a, b) {
					t.Fatalf("filtered results differ after load:\n%v\n%v", a, b)
				}
			}
			for _, id := range []uint64{0, 1, 7, 5999} {
				a, okA := ix.Vector(id)
				b, okB := got.Vector(id)
				if okA != okB || !slices.Equal(a, b) {
					t.Fatalf("Vector(%d) differs after load", id)
				}
			}

			// Saving the loaded index must reproduce the file byte for byte.
			var again bytes.Buffer
			if _, err := got.WriteTo(&again); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(again.Bytes(), data) {
				t.Fatal("re-serialised index differs from the original bytes")
			}

			// Both copies must keep evolving identically.
			extra := randVecs(14, 300, ix.Dim())
			for i, v := range extra {
				id := uint64(len(vecs) + i)
				if err := ix.Add(id, v); err != nil {
					t.Fatal(err)
				}
				if err := got.Add(id, v); err != nil {
					t.Fatal(err)
				}
			}
			for _, q := range extra[:50] {
				if a, b := ix.Search(q, 10, 80, nil), got.Search(q, 10, 80, nil); !slices.Equal(a, b) {
					t.Fatal("results diverge after adding to both copies")
				}
			}
		})
	}
}

func TestSaveLoadEmpty(t *testing.T) {
	var buf bytes.Buffer
	if _, err := New(5, Config{M: 4}).WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	ix, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 0 || ix.Dim() != 5 || ix.Config().M != 4 {
		t.Fatalf("loaded empty index: Len=%d Dim=%d M=%d", ix.Len(), ix.Dim(), ix.Config().M)
	}
	if err := ix.Add(1, make([]float32, 5)); err != nil {
		t.Fatal(err)
	}
	if got := ix.Search(make([]float32, 5), 1, 10, nil); len(got) != 1 {
		t.Fatalf("search after load+add = %v", got)
	}
}

func TestLoadRejectsCorruption(t *testing.T) {
	_, _, data := buildSaved(t, vec.L2)

	mustFail := func(name string, b []byte) {
		t.Helper()
		ix, err := Load(bytes.NewReader(b))
		if err == nil {
			t.Fatalf("%s: Load accepted a damaged stream", name)
		}
		if !errors.Is(err, ErrCorrupt) {
			t.Fatalf("%s: err = %v, want ErrCorrupt", name, err)
		}
		if ix != nil {
			t.Fatalf("%s: Load returned an index alongside an error", name)
		}
	}

	mustFail("empty", nil)
	mustFail("garbage", bytes.Repeat([]byte{0xAB}, 4096))

	cuts := []int{1, 3, headerSize - 1, headerSize, headerSize + 5, len(data) / 2, len(data) - 5, len(data) - 1}
	rng := rand.New(rand.NewSource(15))
	for i := 0; i < 40; i++ {
		cuts = append(cuts, rng.Intn(len(data)))
	}
	for _, n := range cuts {
		mustFail("truncated at "+strconv.Itoa(n), data[:n])
	}

	// Every header byte, the trailer, and a sample of the body.
	flips := []int{len(data) - 1, len(data) - 4}
	for i := 0; i < headerSize; i++ {
		flips = append(flips, i)
	}
	for i := 0; i < 300; i++ {
		flips = append(flips, headerSize+rng.Intn(len(data)-headerSize))
	}
	for _, off := range flips {
		bad := slices.Clone(data)
		bad[off] ^= 1 << uint(rng.Intn(8))
		mustFail("bit flip at "+strconv.Itoa(off), bad)
	}
}

// A file can have valid checksums and still describe a graph that would send
// a search out of bounds. Load has to catch that too.
func TestLoadRejectsInconsistentGraph(t *testing.T) {
	ix := New(2, Config{M: 2})
	for i := 0; i < 50; i++ {
		if err := ix.Add(uint64(i), []float32{float32(i), float32(i * i)}); err != nil {
			t.Fatal(err)
		}
	}
	tamper := func(name string, mutate func(ix *Index)) {
		t.Helper()
		var buf bytes.Buffer
		if _, err := ix.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		victim, err := Load(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		mutate(victim)
		buf.Reset()
		if _, err := victim.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(&buf); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("%s: err = %v, want ErrCorrupt", name, err)
		}
	}
	links0 := func(ix *Index) []uint32 { return (*ix.table.Load())[0].links }

	tamper("neighbour past the end", func(ix *Index) { links0(ix)[1] = 1 << 30 })
	tamper("neighbour count over the limit", func(ix *Index) { links0(ix)[0] = 99 })
	tamper("self loop", func(ix *Index) { links0(ix)[1] = 0 })
	tamper("entry point past the end", func(ix *Index) { ix.entry.Store(packEntry(4000, 0)) })
	tamper("duplicate live id", func(ix *Index) { (*ix.table.Load())[0].nodes[1].id = 0 })
}

type failingWriter struct{ left int }

func (w *failingWriter) Write(p []byte) (int, error) {
	if len(p) > w.left {
		n := w.left
		w.left = 0
		return n, errors.New("disk full")
	}
	w.left -= len(p)
	return len(p), nil
}

func TestWriteToPropagatesError(t *testing.T) {
	ix := New(4, Config{})
	for i, v := range randVecs(16, 100, 4) {
		if err := ix.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	n, err := ix.WriteTo(&failingWriter{left: 100})
	if err == nil || n != 100 {
		t.Fatalf("WriteTo = %d, %v; want 100 and an error", n, err)
	}
}

// TestScale is a quick end-to-end measurement: set HNSW_SCALE=100000 to build that many 128-d vectors with one goroutine per
// core and report build time, recall@10 and single-thread QPS at ef=100.
func TestScale(t *testing.T) {
	n, _ := strconv.Atoi(os.Getenv("HNSW_SCALE"))
	if n == 0 {
		t.Skip("set HNSW_SCALE=<vectors> to run")
	}
	const dim, k, ef, queries = 128, 10, 100, 1000
	vecs := randVecs(1, n, dim)
	ix := New(dim, Config{M: 16, EfConstruction: 200})

	start := time.Now()
	buildParallel(t, ix, vecs)
	build := time.Since(start)
	t.Logf("build: %d x %d-d in %v on %d goroutines (%.0f inserts/s)",
		n, dim, build.Round(time.Millisecond), runtime.GOMAXPROCS(0), float64(n)/build.Seconds())

	qs := randVecs(2, queries, dim)
	got := make([][]Result, queries)
	start = time.Now()
	for i, q := range qs {
		got[i] = ix.Search(q, k, ef, nil)
	}
	elapsed := time.Since(start)

	var sum float64
	for i, q := range qs {
		sum += recall(got[i], bruteForce(vec.L2, vecs, q, k, nil))
	}
	t.Logf("search: recall@%d = %.4f at ef=%d, %.0f QPS single-threaded (%v/query)",
		k, sum/queries, ef, queries/elapsed.Seconds(), (elapsed / queries).Round(time.Microsecond))
}
