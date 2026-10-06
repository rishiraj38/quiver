//go:build !arm64

package vec

// L2Squared returns the squared Euclidean distance between a and b.
func L2Squared(a, b []float32) float32 { return l2SquaredGeneric(a, b) }

// DotProduct returns the inner product of a and b.
func DotProduct(a, b []float32) float32 { return dotGeneric(a, b) }
