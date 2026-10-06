package hnsw

import (
	"sync"
	"sync/atomic"
)

// Nodes live in fixed-size chunks so the arena can grow without moving
// anything a concurrent reader might be looking at. 4096 nodes of 128-d
// float32 is 2 MiB of vectors per chunk.
const (
	chunkShift = 12
	chunkSize  = 1 << chunkShift
	chunkMask  = chunkSize - 1
)

// node is the per-vector metadata. id, level and the upper slice header are
// written once, before the node is reachable, and never change.
type node struct {
	mu      sync.Mutex // guards this node's neighbour lists on every layer
	id      uint64
	upper   []uint32 // layers 1..level, stride slots each: count, then neighbours
	deleted atomic.Bool
	level   uint8
}

// chunk holds chunkSize nodes as three parallel arrays. Layer 0 is where
// every search spends its time, so its lists are stored inline at a fixed
// stride; only the ~1/M of nodes that reach an upper layer pay for a
// separate allocation.
type chunk struct {
	vecs  []float32 // dim floats per node
	links []uint32  // layer 0, stride0 slots per node: count, then neighbours
	nodes []node
}

func (ix *Index) newChunk() *chunk {
	return &chunk{
		vecs:  make([]float32, chunkSize*ix.dim),
		links: make([]uint32, chunkSize*ix.stride0),
		nodes: make([]node, chunkSize),
	}
}

// grow makes sure the chunk for slot exists and returns the current table.
// The caller must hold ix.mu.
func (ix *Index) grow(slot uint32) []*chunk {
	tab := *ix.table.Load()
	if int(slot>>chunkShift) < len(tab) {
		return tab
	}
	next := make([]*chunk, len(tab)+1)
	copy(next, tab)
	next[len(tab)] = ix.newChunk()
	ix.table.Store(&next)
	return next
}

func (ix *Index) vector(tab []*chunk, slot uint32) []float32 {
	off := int(slot&chunkMask) * ix.dim
	return tab[slot>>chunkShift].vecs[off : off+ix.dim : off+ix.dim]
}

// links returns node i's list on the given layer: element 0 is the neighbour
// count and the neighbours follow. The node's lock must be held.
func (ix *Index) links(c *chunk, i uint32, layer int) []uint32 {
	if layer == 0 {
		off := int(i) * ix.stride0
		return c.links[off : off+ix.stride0]
	}
	off := (layer - 1) * ix.stride
	return c.nodes[i].upper[off : off+ix.stride]
}

// neighbours copies slot's list into buf. Copying keeps the critical section
// to a few dozen words; the distance computations that follow run unlocked.
func (ix *Index) neighbours(tab []*chunk, slot uint32, layer int, buf []uint32) []uint32 {
	c := tab[slot>>chunkShift]
	i := slot & chunkMask
	nd := &c.nodes[i]
	nd.mu.Lock()
	l := ix.links(c, i, layer)
	buf = append(buf[:0], l[1:1+l[0]]...)
	nd.mu.Unlock()
	return buf
}

// scratch is the per-operation working memory, pooled so that a search
// allocates nothing but its result slice.
type scratch struct {
	// visited[slot] == epoch means slot was seen by the current search.
	// Bumping the epoch clears the set in O(1); a map or a fresh bitset
	// per query would cost more than the search itself on a large index.
	visited []uint16
	epoch   uint16

	cands minHeap  // frontier, closest first
	best  maxHeap  // current result set, furthest on top
	nbrs  []uint32 // copy of the list being expanded
	out   []cand   // best, sorted closest first
	sel   []cand   // neighbours chosen for the node being inserted
	prune []cand   // candidates while shrinking an overfull list
	kept  []cand   // survivors of that shrink
	query []float32
}

func (ix *Index) getScratch() *scratch {
	if s, ok := ix.pool.Get().(*scratch); ok {
		return s
	}
	return new(scratch)
}

// begin starts a new visited set able to hold slots below n.
func (s *scratch) begin(n int) {
	if len(s.visited) < n {
		// Headroom so a growing index does not reallocate on every chunk.
		s.visited = make([]uint16, n+n/4)
		s.epoch = 0
	}
	s.epoch++
	if s.epoch == 0 {
		// The counter wrapped: stamps from 65536 searches ago would now
		// read as current, so wipe them.
		clear(s.visited)
		s.epoch = 1
	}
}
