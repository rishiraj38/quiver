package hnsw

import (
	"math/rand"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

const (
	benchDim = 128
	benchN   = 50000
)

var (
	benchOnce  sync.Once
	benchIndex *Index
	benchQuery [][]float32
)

// benchSetup builds the shared read-only index once for all search
// benchmarks.
func benchSetup(b *testing.B) (*Index, [][]float32) {
	benchOnce.Do(func() {
		benchIndex = New(benchDim, Config{})
		buildParallel(b, benchIndex, randVecs(1, benchN, benchDim))
		benchQuery = randVecs(2, 1024, benchDim)
	})
	return benchIndex, benchQuery
}

func BenchmarkAdd(b *testing.B) {
	vecs := randVecs(1, b.N, benchDim)
	ix := New(benchDim, Config{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ix.Add(uint64(i), vecs[i]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAddParallel(b *testing.B) {
	vecs := randVecs(1, b.N, benchDim)
	ix := New(benchDim, Config{})
	var next atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := next.Add(1) - 1
			if err := ix.Add(uint64(i), vecs[i]); err != nil {
				b.Error(err)
				return
			}
		}
	})
}

func BenchmarkSearch(b *testing.B) {
	ix, qs := benchSetup(b)
	for _, ef := range []int{10, 50, 100, 200} {
		b.Run("ef="+strconv.Itoa(ef), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ix.Search(qs[i%len(qs)], 10, ef, nil)
			}
		})
	}
}

func BenchmarkSearchParallel(b *testing.B) {
	ix, qs := benchSetup(b)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := rand.Int()
		for pb.Next() {
			ix.Search(qs[i%len(qs)], 10, 100, nil)
			i++
		}
	})
}

func BenchmarkSearchFiltered(b *testing.B) {
	ix, qs := benchSetup(b)
	allow := func(id uint64) bool { return id%20 == 0 }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ix.Search(qs[i%len(qs)], 10, 100, allow)
	}
}
