package pq

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/rishiraj38/quiver/internal/vec"
)

// bruteForce returns the ids (indices into data) of the k nearest vectors.
func bruteForce(data [][]float32, query []float32, k int) []uint64 {
	hits := make([]Hit, len(data))
	for i, v := range data {
		hits[i] = Hit{ID: uint64(i), Dist: vec.L2Squared(query, v)}
	}
	slices.SortFunc(hits, func(a, b Hit) int {
		if a.Dist < b.Dist {
			return -1
		}
		if a.Dist > b.Dist {
			return 1
		}
		return 0
	})
	ids := make([]uint64, k)
	for i := range ids {
		ids[i] = hits[i].ID
	}
	return ids
}

func recall(got []Hit, truth []uint64) float64 {
	found := 0
	for _, h := range got {
		if slices.Contains(truth, h.ID) {
			found++
		}
	}
	return float64(found) / float64(len(truth))
}

func buildFlat(tb testing.TB, data [][]float32, m int) *FlatIndex {
	tb.Helper()
	// A 5k sample is plenty for 256 centroids and keeps -race runs short.
	q := mustTrain(tb, data[:min(len(data), 5000)], m, 15, 1)
	f := NewFlatIndex(q)
	for i, v := range data {
		f.Add(uint64(i), v)
	}
	return f
}

func TestFlatIndexRecall(t *testing.T) {
	const (
		n, dim, m = 20000, 64, 8
		k         = 10
		nq        = 200
	)
	// One draw, split, so queries come from the same mixture as the indexed
	// vectors without being in the index.
	all := lowRank(n+nq, dim, 100, 8, 0.05, 21)
	data, queries := all[:n], all[n:]

	start := time.Now()
	f := buildFlat(t, data, m)
	t.Logf("train on 5000 + encode %d: %v", n, time.Since(start).Round(time.Millisecond))

	if f.Len() != n {
		t.Fatalf("Len = %d, want %d", f.Len(), n)
	}
	raw := n * dim * 4
	t.Logf("raw vectors %d B, index %d B (codes+ids, incl. append slack), codes alone %d B: %.0fx on vectors, %.1fx incl. ids",
		raw, f.Bytes(), n*m, float64(raw)/float64(n*m), float64(raw)/float64(n*(m+8)))
	if f.Bytes() < n*(m+8) {
		t.Fatalf("Bytes = %d, less than the %d the codes and ids occupy", f.Bytes(), n*(m+8))
	}

	exact := func(id uint64) []float32 { return data[id] }
	var plain, reranked4, reranked10 float64
	for _, query := range queries {
		truth := bruteForce(data, query, k)
		plain += recall(f.Search(query, k, 0, nil), truth)
		reranked4 += recall(f.Search(query, k, 4, exact), truth)
		reranked10 += recall(f.Search(query, k, 10, exact), truth)
	}
	plain /= nq
	reranked4 /= nq
	reranked10 /= nq
	t.Logf("recall@%d over %d queries: ADC only %.3f, rerank x4 %.3f, rerank x10 %.3f", k, nq, plain, reranked4, reranked10)

	if plain < 0.5 {
		t.Errorf("ADC-only recall %.3f is too low for the codebooks to be working", plain)
	}
	if reranked4 <= plain {
		t.Errorf("rerank x4 recall %.3f did not improve on ADC-only %.3f", reranked4, plain)
	}
	if reranked10 < reranked4 {
		t.Errorf("rerank x10 recall %.3f is below rerank x4 %.3f", reranked10, reranked4)
	}
	if reranked10 < 0.9 {
		t.Errorf("rerank x10 recall %.3f, want at least 0.9", reranked10)
	}
}

func TestFlatIndexSearchContract(t *testing.T) {
	data := clustered(1000, 16, 8, 0.3, 22)
	f := buildFlat(t, data, 4)
	exact := func(id uint64) []float32 { return data[id] }
	query := data[3]

	t.Run("sorted and sized", func(t *testing.T) {
		for _, rerank := range []int{0, 1, 5} {
			hits := f.Search(query, 10, rerank, exact)
			if len(hits) != 10 {
				t.Fatalf("rerank=%d: %d hits, want 10", rerank, len(hits))
			}
			if !slices.IsSortedFunc(hits, func(a, b Hit) int {
				if a.Dist < b.Dist {
					return -1
				}
				if a.Dist > b.Dist {
					return 1
				}
				return 0
			}) {
				t.Fatalf("rerank=%d: hits not sorted by distance: %v", rerank, hits)
			}
		}
	})

	t.Run("matches a full ADC sort", func(t *testing.T) {
		table := f.q.NewTable(query)
		all := make([]Hit, f.Len())
		for i := range all {
			all[i] = Hit{ID: f.ids[i], Dist: table.Distance(f.codes[i*4 : (i+1)*4])}
		}
		slices.SortFunc(all, func(a, b Hit) int {
			if a.Dist != b.Dist {
				if a.Dist < b.Dist {
					return -1
				}
				return 1
			}
			return 0
		})
		hits := f.Search(query, 25, 0, nil)
		for i, h := range hits {
			if h.Dist != all[i].Dist {
				t.Fatalf("hit %d has distance %g, a full sort gives %g", i, h.Dist, all[i].Dist)
			}
		}
	})

	t.Run("rerank returns exact distances", func(t *testing.T) {
		hits := f.Search(query, 5, 3, exact)
		if hits[0].ID != 3 || hits[0].Dist != 0 {
			t.Fatalf("nearest to an indexed vector is %+v, want id 3 at distance 0", hits[0])
		}
		for _, h := range hits {
			if want := vec.L2Squared(query, data[h.ID]); h.Dist != want {
				t.Fatalf("id %d: distance %g, exact is %g", h.ID, h.Dist, want)
			}
		}
	})

	t.Run("rerank ignored without exact", func(t *testing.T) {
		if a, b := f.Search(query, 10, 8, nil), f.Search(query, 10, 0, nil); !slices.Equal(a, b) {
			t.Fatal("rerank with a nil exact func changed the result")
		}
	})

	t.Run("nil from exact drops the candidate", func(t *testing.T) {
		hits := f.Search(query, 5, 3, func(id uint64) []float32 {
			if id == 3 {
				return nil
			}
			return data[id]
		})
		for _, h := range hits {
			if h.ID == 3 {
				t.Fatal("candidate with no exact vector was returned")
			}
		}
	})

	t.Run("edge sizes", func(t *testing.T) {
		if hits := f.Search(query, 0, 0, nil); len(hits) != 0 {
			t.Fatalf("k=0 returned %d hits", len(hits))
		}
		if hits := f.Search(query, -1, 4, exact); len(hits) != 0 {
			t.Fatalf("k=-1 returned %d hits", len(hits))
		}
		if hits := f.Search(query, 5000, 0, nil); len(hits) != f.Len() {
			t.Fatalf("k > Len returned %d hits, want %d", len(hits), f.Len())
		}
		if hits := f.Search(query, 600, 1<<62, exact); len(hits) != 600 {
			t.Fatalf("huge rerank returned %d hits, want 600", len(hits))
		}
		empty := NewFlatIndex(f.q)
		if hits := empty.Search(query, 10, 4, exact); len(hits) != 0 {
			t.Fatalf("empty index returned %d hits", len(hits))
		}
		if empty.Len() != 0 || empty.Bytes() != 0 {
			t.Fatalf("empty index: Len %d, Bytes %d", empty.Len(), empty.Bytes())
		}
	})
}

func TestFlatIndexConcurrentSearchAndAdd(t *testing.T) {
	data := clustered(3000, 16, 8, 0.3, 23)
	q := mustTrain(t, data[:1000], 4, 5, 1)
	f := NewFlatIndex(q)
	for i, v := range data[:500] {
		f.Add(uint64(i), v)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, v := range data[500:] {
			f.Add(uint64(500+i), v)
		}
	}()
	exact := func(id uint64) []float32 { return data[id] }
	for r := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				hits := f.Search(data[(r*200+i)%len(data)], 10, 3, exact)
				if len(hits) != 10 {
					t.Errorf("got %d hits during concurrent adds, want 10", len(hits))
					return
				}
				_ = f.Len() + f.Bytes()
			}
		}()
	}
	wg.Wait()
	if f.Len() != len(data) {
		t.Fatalf("Len = %d after concurrent adds, want %d", f.Len(), len(data))
	}
}

func BenchmarkFlatSearch(b *testing.B) {
	data := lowRank(20000, 64, 100, 8, 0.05, 21)
	f := buildFlat(b, data, 8)
	exact := func(id uint64) []float32 { return data[id] }
	for _, bc := range []struct {
		name   string
		rerank int
	}{{"adc", 0}, {"rerank10", 10}} {
		b.Run(bc.name, func(b *testing.B) {
			b.ReportAllocs()
			i := 0
			for b.Loop() {
				f.Search(data[i%len(data)], 10, bc.rerank, exact)
				i++
			}
		})
	}
}
