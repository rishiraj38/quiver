// Package pq implements product quantisation (Jégou et al., 2011): a vector
// is split into M sub-vectors and each is replaced by the index of its
// nearest centroid, so a vector costs M bytes instead of 4*dim.
package pq

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/rishiraj38/quiver/internal/vec"
)

// ksub is the number of centroids per sub-space. 256 is what makes a code
// one byte per sub-space, and it lets a byte index a [256]float32 row with
// no bounds check.
const ksub = 256

const defaultIters = 25

// Quantizer holds M codebooks of 256 centroids each. Squared L2 only.
//
// A trained Quantizer is immutable, so it is safe for concurrent use.
type Quantizer struct {
	dim  int
	m    int
	dsub int // dim / m

	// codebooks is m*ksub*dsub floats: centroid c of sub-space s starts at
	// (s*ksub+c)*dsub. One allocation keeps the 256 candidates that Encode
	// scans for a sub-space adjacent in memory.
	codebooks []float32
}

// Train learns the codebooks with k-means (k-means++ seeding) over a sample.
// dim must be divisible by m. It needs at least 256 training vectors.
//
// iters caps the Lloyd iterations per sub-space; iters <= 0 selects a default.
// The result depends only on the sample, m, iters and seed, not on the number
// of cores.
func Train(sample [][]float32, m, iters int, seed int64) (*Quantizer, error) {
	if len(sample) < ksub {
		return nil, fmt.Errorf("pq: need at least %d training vectors, got %d", ksub, len(sample))
	}
	dim := len(sample[0])
	if dim == 0 {
		return nil, errors.New("pq: training vectors are empty")
	}
	if m <= 0 || dim%m != 0 {
		return nil, fmt.Errorf("pq: dimension %d is not divisible by m=%d", dim, m)
	}
	for i, v := range sample {
		if len(v) != dim {
			return nil, fmt.Errorf("pq: training vector %d has dimension %d, want %d", i, len(v), dim)
		}
	}
	if iters <= 0 {
		iters = defaultIters
	}

	q := &Quantizer{
		dim:       dim,
		m:         m,
		dsub:      dim / m,
		codebooks: make([]float32, ksub*dim), // m * ksub * dim/m
	}

	// Sub-spaces are independent problems, so they are the unit of
	// parallelism. Each one owns its RNG stream and its slice of the
	// codebook, which is what keeps the output independent of scheduling.
	workers := min(m, runtime.GOMAXPROCS(0))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			km := newKMeans(len(sample), q.dsub)
			for {
				s := int(next.Add(1)) - 1
				if s >= m {
					return
				}
				km.load(sample, s)
				km.run(q.codebook(s), iters, seed, s)
			}
		}()
	}
	wg.Wait()
	return q, nil
}

func (q *Quantizer) Dim() int { return q.dim }

// M is the code length in bytes.
func (q *Quantizer) M() int { return q.m }

// codebook returns the ksub*dsub centroids of sub-space s.
func (q *Quantizer) codebook(s int) []float32 {
	w := ksub * q.dsub
	return q.codebooks[s*w : (s+1)*w : (s+1)*w]
}

// Encode appends the M-byte code for v to dst and returns it.
// It panics if len(v) != Dim().
func (q *Quantizer) Encode(dst []byte, v []float32) []byte {
	if len(v) != q.dim {
		panic(fmt.Sprintf("pq: Encode: vector has dimension %d, want %d", len(v), q.dim))
	}
	for s := 0; s < q.m; s++ {
		c, _ := nearest(q.codebook(s), q.dsub, v[s*q.dsub:(s+1)*q.dsub])
		dst = append(dst, byte(c))
	}
	return dst
}

// Decode appends the reconstructed vector for code to dst and returns it.
// It panics if len(code) != M().
func (q *Quantizer) Decode(dst []float32, code []byte) []float32 {
	if len(code) != q.m {
		panic(fmt.Sprintf("pq: Decode: code has length %d, want %d", len(code), q.m))
	}
	for s, c := range code {
		off := (s*ksub + int(c)) * q.dsub
		dst = append(dst, q.codebooks[off:off+q.dsub]...)
	}
	return dst
}

// nearest returns the index of the centroid in cb closest to x and its
// squared distance. Ties go to the lowest index.
func nearest(cb []float32, dsub int, x []float32) (int, float32) {
	best, bestD := 0, vec.L2Squared(x, cb[:dsub])
	for c := 1; c*dsub < len(cb); c++ {
		if d := vec.L2Squared(x, cb[c*dsub:(c+1)*dsub]); d < bestD {
			best, bestD = c, d
		}
	}
	return best, bestD
}

// Table is the per-query lookup table for asymmetric distance computation:
// the query stays exact, stored vectors are compared through their codes.
type Table struct {
	// One row per sub-space. A slice of arrays is still a single contiguous
	// allocation, and indexing a [256]float32 with a byte needs no bounds
	// check.
	rows [][ksub]float32
}

// NewTable precomputes the M x 256 sub-distances for query.
// It panics if len(query) != Dim().
func (q *Quantizer) NewTable(query []float32) *Table {
	if len(query) != q.dim {
		panic(fmt.Sprintf("pq: NewTable: query has dimension %d, want %d", len(query), q.dim))
	}
	t := &Table{rows: make([][ksub]float32, q.m)}
	for s := range t.rows {
		sub := query[s*q.dsub : (s+1)*q.dsub]
		cb := q.codebook(s)
		row := &t.rows[s]
		for c := range row {
			row[c] = vec.L2Squared(sub, cb[c*q.dsub:(c+1)*q.dsub])
		}
	}
	return t
}

// Distance returns the approximate squared L2 distance to an encoded vector.
// It panics if code is shorter than M bytes.
func (t *Table) Distance(code []byte) float32 {
	rows := t.rows
	// Reslicing to len(rows) proves code[i] is in range wherever rows[i] is,
	// and a byte cannot overrun a [256]float32, so what is left is one check
	// per group of four lookups.
	code = code[:len(rows)]

	// Four accumulators so the adds do not form one dependency chain. This
	// loop runs once per stored vector per query. Measured against the plain
	// loop it is level at M=8 and roughly 20% faster at M=32.
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(rows); i += 4 {
		r, c := rows[i:i+4:i+4], code[i:i+4:i+4]
		s0 += r[0][c[0]]
		s1 += r[1][c[1]]
		s2 += r[2][c[2]]
		s3 += r[3][c[3]]
	}
	for ; i < len(rows); i++ {
		s0 += rows[i][code[i]]
	}
	return s0 + s1 + s2 + s3
}
