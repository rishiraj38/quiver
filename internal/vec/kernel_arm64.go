package vec

// NEON is part of the arm64 baseline, so no feature detection is needed.

//go:noescape
func l2SquaredNEON(a, b *float32, n int) float32

//go:noescape
func dotNEON(a, b *float32, n int) float32

// L2Squared returns the squared Euclidean distance between a and b.
func L2Squared(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	b = b[:len(a)]
	return l2SquaredNEON(&a[0], &b[0], len(a))
}

// DotProduct returns the inner product of a and b.
func DotProduct(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	b = b[:len(a)]
	return dotNEON(&a[0], &b[0], len(a))
}
