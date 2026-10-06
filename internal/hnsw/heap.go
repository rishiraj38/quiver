package hnsw

// cand is a node paired with its distance to whatever is being searched for.
type cand struct {
	dist float32
	id   uint32
}

// The two heaps below are plain binary heaps over []cand. container/heap
// would box every element in an interface and call Less and Swap through a
// method table; in the search inner loop that overhead is measurable.

// minHeap keeps the closest candidate at index 0.
type minHeap []cand

func (h *minHeap) push(c cand) {
	a := append(*h, c)
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist <= c.dist {
			break
		}
		a[i] = a[p]
		i = p
	}
	a[i] = c
	*h = a
}

func (h *minHeap) pop() cand {
	a := *h
	top := a[0]
	n := len(a) - 1
	last := a[n]
	a = a[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		if r := l + 1; r < n && a[r].dist < a[l].dist {
			l = r
		}
		if last.dist <= a[l].dist {
			break
		}
		a[i] = a[l]
		i = l
	}
	if n > 0 {
		a[i] = last
	}
	*h = a
	return top
}

// maxHeap keeps the furthest candidate at index 0, so a bounded result set
// can evict its worst member in O(log n).
type maxHeap []cand

func (h *maxHeap) push(c cand) {
	a := append(*h, c)
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist >= c.dist {
			break
		}
		a[i] = a[p]
		i = p
	}
	a[i] = c
	*h = a
}

func (h *maxHeap) pop() cand {
	a := *h
	top := a[0]
	n := len(a) - 1
	last := a[n]
	a = a[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		if r := l + 1; r < n && a[r].dist > a[l].dist {
			l = r
		}
		if last.dist >= a[l].dist {
			break
		}
		a[i] = a[l]
		i = l
	}
	if n > 0 {
		a[i] = last
	}
	*h = a
	return top
}

// drainSorted empties the heap into dst, closest first, reusing dst's
// storage.
func (h *maxHeap) drainSorted(dst []cand) []cand {
	n := len(*h)
	if cap(dst) < n {
		dst = make([]cand, n)
	}
	dst = dst[:n]
	for i := n - 1; i >= 0; i-- {
		dst[i] = h.pop()
	}
	return dst
}
