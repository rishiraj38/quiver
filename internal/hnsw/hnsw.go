// Package hnsw implements a Hierarchical Navigable Small World graph index
// (Malkov & Yashunin, 2016).
package hnsw

import (
	"cmp"
	"errors"
	"math"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/rishiraj38/quiver/internal/vec"
)

// Config tunes graph construction. Zero values fall back to the defaults.
type Config struct {
	M              int        // max neighbours per node on layers > 0 (layer 0 keeps 2*M); default 16
	EfConstruction int        // candidate list size while inserting; default 200
	Metric         vec.Metric // default vec.L2
	Seed           int64      // level generator seed; 0 means a fixed default so builds are reproducible
}

// Result is one search hit. Dist follows vec.Metric: smaller is closer.
type Result struct {
	ID   uint64
	Dist float32
}

var (
	ErrDimension = errors.New("hnsw: vector has wrong dimension")
	ErrExists    = errors.New("hnsw: id already present")
	// ErrFull is returned by Add once 2^32-1 slots (tombstones included)
	// have been handed out.
	ErrFull = errors.New("hnsw: index is full")
)

const (
	defaultM              = 16
	defaultEfConstruction = 200
	defaultSeed           = 1

	// Upper bounds on what New accepts. They exist so that Load can refuse
	// a header asking for absurd allocations, and anything New builds is
	// guaranteed to be loadable.
	maxM   = 4096
	maxDim = 1 << 16

	// With mL = 1/ln(M) a level above 31 has probability below 1e-9 even
	// for M=2, so the cap never shapes the distribution in practice.
	maxLevel = 31

	noEntry = ^uint64(0)
)

// Index is safe for concurrent use: many Search calls may run alongside
// Add and Delete.
//
// Locking, from coarsest to finest:
//
//	saveMu  held shared by Add for the whole insert and exclusively by
//	        WriteTo, so a snapshot never sees a half-linked node.
//	mu      guards ids, count and live, and serialises slot allocation
//	        and entry-point updates. Held only for a few hundred
//	        nanoseconds per Add.
//	node.mu guards one node's neighbour lists. At most one is held at a
//	        time, so there is no lock ordering to get wrong.
//
// Search takes neither saveMu nor mu.
type Index struct {
	dim       int
	cfg       Config
	dist      vec.DistFunc
	levelMult float64
	stride0   int // layer-0 slots per node: count + 2M neighbours
	stride    int // upper-layer slots per node and layer: count + M neighbours

	saveMu sync.RWMutex

	mu    sync.RWMutex
	ids   map[uint64]uint32 // live external id -> slot
	count uint32            // slots handed out, tombstones included
	live  int

	// table is copy-on-write: growing the arena publishes a new, longer
	// slice. Chunks themselves never move, so a reader holding an old
	// table still sees valid memory for every slot it knows about.
	table atomic.Pointer[[]*chunk]

	// entry packs the entry point slot (low 32 bits) with its level (high
	// 32 bits) so readers get a consistent pair from one atomic load.
	entry atomic.Uint64

	pool sync.Pool // *scratch
}

// New returns an empty index for vectors of the given dimension. It panics
// if dim is outside [1, 65536]. M is clamped to [2, 4096].
func New(dim int, cfg Config) *Index {
	if dim < 1 || dim > maxDim {
		panic("hnsw: dimension out of range")
	}
	if cfg.M <= 0 {
		cfg.M = defaultM
	}
	// M=1 would make the level multiplier 1/ln(1) infinite.
	cfg.M = min(max(cfg.M, 2), maxM)
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = defaultEfConstruction
	}
	if cfg.Seed == 0 {
		cfg.Seed = defaultSeed
	}
	ix := &Index{
		dim:       dim,
		cfg:       cfg,
		dist:      cfg.Metric.Func(),
		levelMult: 1 / math.Log(float64(cfg.M)),
		stride0:   2*cfg.M + 1,
		stride:    cfg.M + 1,
		ids:       make(map[uint64]uint32),
	}
	ix.table.Store(new([]*chunk))
	ix.entry.Store(noEntry)
	return ix
}

func packEntry(slot uint32, level int) uint64 { return uint64(level)<<32 | uint64(slot) }

func unpackEntry(e uint64) (slot uint32, level int) { return uint32(e), int(e >> 32) }

// levelFor draws the level of the node in the given slot from the
// exponential distribution floor(-ln(U) * mL). U comes from hashing
// (seed, slot) with splitmix64 rather than from a stateful generator: there
// is nothing to lock, and an index that was saved and loaded assigns the same
// levels to later inserts as the original would have.
func (ix *Index) levelFor(slot uint32) int {
	x := uint64(ix.cfg.Seed) + (uint64(slot)+1)*0x9E3779B97F4A7C15
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	u := (float64(x>>11) + 1) / (1 << 53) // uniform in (0, 1], so Log is finite
	return min(int(-math.Log(u)*ix.levelMult), maxLevel)
}

// Add inserts v under id. The vector is copied (and normalised when the
// metric is Cosine). It returns ErrExists if id is live and ErrDimension on
// a length mismatch. Re-adding an id that was deleted is allowed.
func (ix *Index) Add(id uint64, v []float32) error {
	if len(v) != ix.dim {
		return ErrDimension
	}
	ix.saveMu.RLock()
	defer ix.saveMu.RUnlock()

	// The node is fully initialised before mu is released. From then on its
	// id, level and vector are immutable, which is what lets searches read
	// them without a lock.
	ix.mu.Lock()
	if _, ok := ix.ids[id]; ok {
		ix.mu.Unlock()
		return ErrExists
	}
	if ix.count == math.MaxUint32 {
		ix.mu.Unlock()
		return ErrFull
	}
	slot := ix.count
	tab := ix.grow(slot)
	level := ix.levelFor(slot)
	nd := &tab[slot>>chunkShift].nodes[slot&chunkMask]
	nd.id = id
	nd.level = uint8(level)
	if level > 0 {
		nd.upper = make([]uint32, level*ix.stride)
	}
	q := ix.vector(tab, slot)
	copy(q, v)
	if ix.cfg.Metric == vec.Cosine {
		vec.Normalize(q)
	}
	ix.ids[id] = slot
	ix.count++
	ix.live++
	entry := ix.entry.Load()
	if entry == noEntry {
		ix.entry.Store(packEntry(slot, level))
		ix.mu.Unlock()
		return nil
	}
	ix.mu.Unlock()

	s := ix.getScratch()
	defer ix.pool.Put(s)

	epSlot, top := unpackEntry(entry)
	cur := cand{dist: ix.dist(q, ix.vector(tab, epSlot)), id: epSlot}
	for l := top; l > level; l-- {
		cur = ix.greedy(s, tab, q, cur, l)
	}
	for l := min(level, top); l >= 0; l-- {
		s.begin(len(tab) << chunkShift)
		// A concurrent insert that descended through this node on a higher
		// layer may already have linked to it down here. Without this the
		// search could find the node itself and pick it as its own neighbour.
		s.visited[slot] = s.epoch
		ix.searchLayer(s, tab, q, cur, ix.cfg.EfConstruction, l, nil, false)
		s.out = s.best.drainSorted(s.out)
		cur = s.out[0]
		// The paper selects M neighbours on every layer and only lets
		// back-links fill layer 0 up to 2M. Selecting up to the layer's
		// full capacity straight away (as FAISS does) measured about 4.5
		// points more recall@10 at the same ef on uniform 64-d data, for
		// no extra memory since the slots are reserved either way.
		capacity := ix.stride - 1
		if l == 0 {
			capacity = ix.stride0 - 1
		}
		s.sel = ix.selectNeighbours(tab, s.out, capacity, s.sel)
		for _, n := range s.sel {
			ix.connect(s, tab, slot, n.id, n.dist, l)
			ix.connect(s, tab, n.id, slot, n.dist, l)
		}
	}

	if level > top {
		// Two inserts can both outgrow the old top level; only the taller
		// one (or the first, on a tie) becomes the entry point. The loser
		// keeps empty lists on its extra layers, which costs nothing: it is
		// still linked on every layer the graph had when it was inserted.
		ix.mu.Lock()
		if _, t := unpackEntry(ix.entry.Load()); level > t {
			ix.entry.Store(packEntry(slot, level))
		}
		ix.mu.Unlock()
	}
	return nil
}

// Delete tombstones id. Deleted nodes stay in the graph as routing hops but
// are never returned. It reports whether id was live.
func (ix *Index) Delete(id uint64) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	slot, ok := ix.ids[id]
	if !ok {
		return false
	}
	delete(ix.ids, id)
	tab := *ix.table.Load()
	tab[slot>>chunkShift].nodes[slot&chunkMask].deleted.Store(true)
	ix.live--
	return true
}

// Search returns up to k nearest live vectors to q, closest first. ef is the
// layer-0 beam width (raised to k if smaller). allow, when non-nil, restricts
// results to ids for which it returns true; rejected nodes are still
// traversed so a selective filter does not disconnect the graph.
//
// A query of the wrong dimension, or k <= 0, yields nil. allow is called
// with no index lock held.
func (ix *Index) Search(q []float32, k, ef int, allow func(id uint64) bool) []Result {
	if k <= 0 || len(q) != ix.dim {
		return nil
	}
	// Order matters: the table is published before the entry point that
	// lives in it, so loading the entry first guarantees the table read
	// afterwards contains it.
	entry := ix.entry.Load()
	if entry == noEntry {
		return nil
	}
	tab := *ix.table.Load()
	ef = max(ef, k)

	s := ix.getScratch()
	defer ix.pool.Put(s)

	if ix.cfg.Metric == vec.Cosine {
		s.query = append(s.query[:0], q...)
		vec.Normalize(s.query)
		q = s.query
	}

	epSlot, top := unpackEntry(entry)
	cur := cand{dist: ix.dist(q, ix.vector(tab, epSlot)), id: epSlot}
	for l := top; l > 0; l-- {
		cur = ix.greedy(s, tab, q, cur, l)
	}
	s.begin(len(tab) << chunkShift)
	ix.searchLayer(s, tab, q, cur, ef, 0, allow, true)
	s.out = s.best.drainSorted(s.out)

	n := min(k, len(s.out))
	if n == 0 {
		return nil
	}
	res := make([]Result, n)
	for i, c := range s.out[:n] {
		res[i] = Result{ID: tab[c.id>>chunkShift].nodes[c.id&chunkMask].id, Dist: c.dist}
	}
	return res
}

// greedy walks one upper layer, moving to whichever neighbour is closer to q
// until none is. It is a beam search of width one, so it needs no visited
// set: the distance to q strictly decreases on every move.
func (ix *Index) greedy(s *scratch, tab []*chunk, q []float32, cur cand, layer int) cand {
	limit := len(tab) << chunkShift
	for improved := true; improved; {
		improved = false
		s.nbrs = ix.neighbours(tab, cur.id, layer, s.nbrs)
		for _, n := range s.nbrs {
			if int(n) >= limit {
				continue
			}
			if d := ix.dist(q, ix.vector(tab, n)); d < cur.dist {
				cur = cand{dist: d, id: n}
				improved = true
			}
		}
	}
	return cur
}

// searchLayer is the ef-bounded best-first search of the paper's Algorithm 2.
// It leaves the (up to) ef closest admissible nodes in s.best. The caller
// must have called s.begin for this tab.
//
// With filter set, tombstoned and disallowed nodes are expanded like any
// other but never enter s.best. Because the stop condition waits for ef
// admissible hits, a selective filter widens the search instead of starving
// it; the price is that a filter matching nothing walks the whole graph.
func (ix *Index) searchLayer(s *scratch, tab []*chunk, q []float32, ep cand, ef, layer int, allow func(uint64) bool, filter bool) {
	limit := len(tab) << chunkShift
	epoch := s.epoch
	s.cands = s.cands[:0]
	s.best = s.best[:0]

	s.visited[ep.id] = epoch
	s.cands.push(ep)
	if !filter || ix.admit(tab, ep.id, allow) {
		s.best.push(ep)
	}

	for len(s.cands) > 0 {
		c := s.cands.pop()
		if len(s.best) >= ef && c.dist > s.best[0].dist {
			break
		}
		s.nbrs = ix.neighbours(tab, c.id, layer, s.nbrs)
		for _, n := range s.nbrs {
			// Slots at or past limit were allocated after this search took
			// its table snapshot; treating them as not yet inserted keeps
			// every array access inside the snapshot.
			if int(n) >= limit || s.visited[n] == epoch {
				continue
			}
			s.visited[n] = epoch
			d := ix.dist(q, ix.vector(tab, n))
			if len(s.best) >= ef && d >= s.best[0].dist {
				continue
			}
			s.cands.push(cand{dist: d, id: n})
			if filter && !ix.admit(tab, n, allow) {
				continue
			}
			s.best.push(cand{dist: d, id: n})
			if len(s.best) > ef {
				s.best.pop()
			}
		}
	}
}

func (ix *Index) admit(tab []*chunk, slot uint32, allow func(uint64) bool) bool {
	nd := &tab[slot>>chunkShift].nodes[slot&chunkMask]
	if nd.deleted.Load() {
		return false
	}
	return allow == nil || allow(nd.id)
}

// selectNeighbours is the paper's Algorithm 4 without its two optional
// extensions (the same variant hnswlib ships). cands must be sorted closest
// first, with dist measured to the base node. A candidate is kept only if it
// is closer to the base than to every neighbour already kept, which spreads
// the links across directions instead of spending them all on one tight
// cluster. That diversity is what keeps the graph navigable on clustered
// data, where plain closest-M leaves clusters connected only to themselves.
func (ix *Index) selectNeighbours(tab []*chunk, cands []cand, m int, dst []cand) []cand {
	dst = dst[:0]
	for _, c := range cands {
		if len(dst) == m {
			break
		}
		cv := ix.vector(tab, c.id)
		keep := true
		for _, r := range dst {
			if ix.dist(cv, ix.vector(tab, r.id)) < c.dist {
				keep = false
				break
			}
		}
		if keep {
			dst = append(dst, c)
		}
	}
	return dst
}

// connect adds the directed edge dst -> src on the given layer, where d is
// the distance between the two. If dst's list is full it is rebuilt by
// running the selection heuristic over the old neighbours plus src, so src
// may itself be the one dropped.
func (ix *Index) connect(s *scratch, tab []*chunk, dst, src uint32, d float32, layer int) {
	c := tab[dst>>chunkShift]
	i := dst & chunkMask
	nd := &c.nodes[i]
	nd.mu.Lock()
	defer nd.mu.Unlock()

	l := ix.links(c, i, layer)
	n := int(l[0])
	// Two concurrent inserts can each select the other, so the edge may
	// already be there.
	if slices.Contains(l[1:1+n], src) {
		return
	}
	if n < len(l)-1 {
		l[1+n] = src
		l[0]++
		return
	}

	// dst's neighbours may have been allocated after the caller took its
	// snapshot. Whoever linked them did so under nd.mu, after publishing
	// their chunk, so a table loaded now is guaranteed to cover them.
	tab = *ix.table.Load()
	dv := ix.vector(tab, dst)
	s.prune = append(s.prune[:0], cand{dist: d, id: src})
	for _, nb := range l[1 : 1+n] {
		s.prune = append(s.prune, cand{dist: ix.dist(dv, ix.vector(tab, nb)), id: nb})
	}
	slices.SortFunc(s.prune, func(a, b cand) int { return cmp.Compare(a.dist, b.dist) })
	s.kept = ix.selectNeighbours(tab, s.prune, len(l)-1, s.kept)

	l[0] = uint32(len(s.kept))
	for j, k := range s.kept {
		l[1+j] = k.id
	}
	// Unused slots stay zero so the serialised form is a function of the
	// graph alone.
	clear(l[1+len(s.kept):])
}

// Vector returns a copy of the stored vector for a live id. Under the Cosine
// metric that is the normalised vector, not the one originally passed in.
func (ix *Index) Vector(id uint64) ([]float32, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	slot, ok := ix.ids[id]
	if !ok {
		return nil, false
	}
	return slices.Clone(ix.vector(*ix.table.Load(), slot)), true
}

// Len is the number of live vectors.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.live
}

// Dim is the vector dimension the index was created with.
func (ix *Index) Dim() int { return ix.dim }

// Config returns the configuration with defaults applied.
func (ix *Index) Config() Config { return ix.cfg }
