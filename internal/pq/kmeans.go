package pq

import (
	"math/rand/v2"

	"github.com/rishiraj38/quiver/internal/vec"
)

// kmeans is the scratch space for clustering one sub-space. A worker reuses
// it across the sub-spaces it trains.
type kmeans struct {
	n, dsub int
	points  []float32 // n*dsub, the sub-vectors copied out of the sample
	assign  []int32   // cluster of each point
	dist    []float32 // squared distance of each point to its centroid
	counts  []int
	sums    []float64 // ksub*dsub; float64 so large clusters do not lose precision
}

func newKMeans(n, dsub int) *kmeans {
	return &kmeans{
		n:      n,
		dsub:   dsub,
		points: make([]float32, n*dsub),
		assign: make([]int32, n),
		dist:   make([]float32, n),
		counts: make([]int, ksub),
		sums:   make([]float64, ksub*dsub),
	}
}

// load copies sub-space s of every sample into one flat buffer. The copy is
// paid once; the assignment step then walks memory sequentially instead of
// chasing a slice header per point on every iteration.
func (km *kmeans) load(sample [][]float32, s int) {
	for i, v := range sample {
		copy(km.points[i*km.dsub:(i+1)*km.dsub], v[s*km.dsub:(s+1)*km.dsub])
	}
}

func (km *kmeans) point(i int) []float32 {
	return km.points[i*km.dsub : (i+1)*km.dsub]
}

// run writes ksub centroids into cb.
func (km *kmeans) run(cb []float32, iters int, seed int64, s int) {
	// The sub-space index selects the PCG stream, so sub-spaces get
	// independent sequences from one user-facing seed.
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(s)))
	km.seed(cb, rng)

	for i := range km.assign {
		km.assign[i] = -1
	}
	for range iters {
		changed := km.assignAll(cb)
		reseeded := km.fillEmpty()
		if changed == 0 && !reseeded {
			break
		}
		km.update(cb)
	}
}

// seed is k-means++: each new centroid is a data point drawn with
// probability proportional to its squared distance from the centroids chosen
// so far, which spreads the seeds out and avoids most bad local minima.
func (km *kmeans) seed(cb []float32, rng *rand.Rand) {
	d := km.dsub
	copy(cb[:d], km.point(rng.IntN(km.n)))

	minDist := km.dist
	var total float64
	for i := range minDist {
		minDist[i] = vec.L2Squared(km.point(i), cb[:d])
		total += float64(minDist[i])
	}

	for c := 1; c < ksub; c++ {
		pick := km.n - 1
		if total > 0 {
			target := rng.Float64() * total
			var acc float64
			for i, w := range minDist {
				acc += float64(w)
				if acc > target {
					pick = i
					break
				}
			}
		} else {
			// Every point already coincides with a centroid (fewer than
			// 256 distinct sub-vectors). Any choice is as good as another.
			pick = rng.IntN(km.n)
		}

		centroid := cb[c*d : (c+1)*d]
		copy(centroid, km.point(pick))

		total = 0
		for i := range minDist {
			if nd := vec.L2Squared(km.point(i), centroid); nd < minDist[i] {
				minDist[i] = nd
			}
			total += float64(minDist[i])
		}
	}
}

// assignAll moves every point to its nearest centroid and reports how many
// points changed cluster.
func (km *kmeans) assignAll(cb []float32) (changed int) {
	clear(km.counts)
	for i := 0; i < km.n; i++ {
		c, dist := nearest(cb, km.dsub, km.point(i))
		if km.assign[i] != int32(c) {
			km.assign[i] = int32(c)
			changed++
		}
		km.dist[i] = dist
		km.counts[c]++
	}
	return changed
}

// fillEmpty gives every empty cluster the point that is currently worst
// served: the one farthest from its own centroid. Leaving a cluster empty
// would waste one of the 256 codes for good.
func (km *kmeans) fillEmpty() (reseeded bool) {
	for c, cnt := range km.counts {
		if cnt != 0 {
			continue
		}
		// Only take from clusters with a point to spare, otherwise the
		// donor becomes the next empty cluster. With n >= ksub one always
		// exists while any cluster is empty.
		far := -1
		for i, d := range km.dist {
			if km.counts[km.assign[i]] > 1 && (far < 0 || d > km.dist[far]) {
				far = i
			}
		}
		if far < 0 {
			return reseeded
		}
		km.counts[km.assign[far]]--
		km.assign[far] = int32(c)
		km.counts[c] = 1
		km.dist[far] = 0 // so the next empty cluster picks a different point
		reseeded = true
	}
	return reseeded
}

// update recomputes each centroid as the mean of its points.
func (km *kmeans) update(cb []float32) {
	d := km.dsub
	clear(km.sums)
	for i, c := range km.assign {
		sum := km.sums[int(c)*d : (int(c)+1)*d]
		for j, x := range km.point(i) {
			sum[j] += float64(x)
		}
	}
	for c, cnt := range km.counts {
		if cnt == 0 {
			continue
		}
		inv := 1 / float64(cnt)
		for j := c * d; j < (c+1)*d; j++ {
			cb[j] = float32(km.sums[j] * inv)
		}
	}
}
