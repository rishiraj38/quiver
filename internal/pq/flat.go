package pq

import (
	"cmp"
	"slices"
	"sync"

	"github.com/rishiraj38/quiver/internal/vec"
)

// Hit is one search result. Dist is a squared L2 distance: approximate
// unless the search re-ranked.
type Hit struct {
	ID   uint64
	Dist float32
}

// FlatIndex stores only PQ codes and answers queries by scanning them with
// an ADC table, then optionally re-ranking the best candidates exactly.
//
// It is safe for concurrent Search alongside a single writer.
type FlatIndex struct {
	q *Quantizer

	mu    sync.RWMutex
	codes []byte // Len()*M bytes; vector i occupies codes[i*M:(i+1)*M]
	ids   []uint64
}

func NewFlatIndex(q *Quantizer) *FlatIndex {
	return &FlatIndex{q: q}
}

// Add encodes v and stores its code under id. It panics if len(v) does not
// match the quantizer's dimension.
func (f *FlatIndex) Add(id uint64, v []float32) {
	// Encoding is the expensive part (M*256 distance evaluations) and needs
	// no shared state, so it happens before the lock is taken. The array
	// keeps the temporary code off the heap for any realistic M.
	var buf [64]byte
	code := f.q.Encode(buf[:0], v)

	f.mu.Lock()
	f.codes = append(f.codes, code...)
	f.ids = append(f.ids, id)
	f.mu.Unlock()
}

func (f *FlatIndex) Len() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.ids)
}

// Bytes is the memory held by codes and ids.
func (f *FlatIndex) Bytes() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return cap(f.codes) + 8*cap(f.ids)
}

// Search returns the k nearest by approximate distance. If rerank > 1 and
// exact != nil, it takes the top k*rerank by ADC distance, recomputes their
// true squared-L2 distance via exact(id) and returns the best k of those.
//
// Results are sorted by ascending distance. A candidate for which exact
// returns nil is dropped, so a re-ranked search can return fewer than k hits.
func (f *FlatIndex) Search(query []float32, k, rerank int, exact func(id uint64) []float32) []Hit {
	if k <= 0 {
		return nil
	}
	table := f.q.NewTable(query)
	m := f.q.m

	f.mu.RLock()
	n := len(f.ids)
	want := min(k, n)
	doRerank := rerank > 1 && exact != nil
	if doRerank {
		// Compared by division so k*rerank cannot overflow.
		want = n
		if rerank <= n/k {
			want = k * rerank
		}
	}
	// top is a max-heap of the best `want` seen so far: its root is the
	// worst of them, which is the only one a new code has to beat.
	top := make([]Hit, 0, want)
	for i := 0; i < n && want > 0; i++ {
		d := table.Distance(f.codes[i*m : (i+1)*m])
		if len(top) < want {
			top = append(top, Hit{ID: f.ids[i], Dist: d})
			siftUp(top, len(top)-1)
		} else if d < top[0].Dist {
			top[0] = Hit{ID: f.ids[i], Dist: d}
			siftDown(top, 0)
		}
	}
	// Released before calling exact: the callback belongs to the caller and
	// may block or take its own locks.
	f.mu.RUnlock()

	if doRerank {
		kept := top[:0]
		for _, h := range top {
			if v := exact(h.ID); v != nil {
				kept = append(kept, Hit{ID: h.ID, Dist: vec.L2Squared(query, v)})
			}
		}
		top = kept
	}

	// ID breaks ties so equal distances come back in a stable order.
	slices.SortFunc(top, func(a, b Hit) int {
		if c := cmp.Compare(a.Dist, b.Dist); c != 0 {
			return c
		}
		return cmp.Compare(a.ID, b.ID)
	})
	if len(top) > k {
		top = top[:k]
	}
	return top
}

// A hand-rolled heap instead of container/heap: that one goes through an
// interface per comparison, which is measurable in a loop this hot.

func siftUp(h []Hit, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].Dist >= h[i].Dist {
			return
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

func siftDown(h []Hit, i int) {
	for {
		largest := i
		if l := 2*i + 1; l < len(h) && h[l].Dist > h[largest].Dist {
			largest = l
		}
		if r := 2*i + 2; r < len(h) && h[r].Dist > h[largest].Dist {
			largest = r
		}
		if largest == i {
			return
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}
