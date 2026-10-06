// Package vec holds the distance kernels shared by every index.
package vec

import (
	"fmt"
	"math"
)

// Metric selects how two vectors are compared. Every metric is exposed as a
// distance: smaller means closer.
type Metric uint8

const (
	L2     Metric = iota // squared Euclidean distance
	Cosine               // 1 - cosine similarity; vectors must be unit length
	Dot                  // negated inner product
)

func (m Metric) String() string {
	switch m {
	case L2:
		return "l2"
	case Cosine:
		return "cosine"
	case Dot:
		return "dot"
	}
	return fmt.Sprintf("metric(%d)", uint8(m))
}

// ParseMetric is the inverse of Metric.String.
func ParseMetric(s string) (Metric, error) {
	switch s {
	case "l2", "":
		return L2, nil
	case "cosine":
		return Cosine, nil
	case "dot":
		return Dot, nil
	}
	return 0, fmt.Errorf("vec: unknown metric %q", s)
}

// DistFunc compares two vectors of equal length.
type DistFunc func(a, b []float32) float32

// Func returns the distance kernel for m. For Cosine the caller is expected
// to have normalised both vectors (see Normalize).
func (m Metric) Func() DistFunc {
	switch m {
	case Cosine:
		return func(a, b []float32) float32 { return 1 - DotProduct(a, b) }
	case Dot:
		return func(a, b []float32) float32 { return -DotProduct(a, b) }
	}
	return L2Squared
}

// l2SquaredGeneric is the portable kernel. The loop is unrolled four ways so
// the compiler keeps four independent accumulators and drops the bounds
// checks.
func l2SquaredGeneric(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		x, y := a[i:i+4:i+4], b[i:i+4:i+4]
		d0 := x[0] - y[0]
		d1 := x[1] - y[1]
		d2 := x[2] - y[2]
		d3 := x[3] - y[3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return s0 + s1 + s2 + s3
}

func dotGeneric(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		x, y := a[i:i+4:i+4], b[i:i+4:i+4]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// Normalize scales v to unit length in place. A zero vector is left as is.
func Normalize(v []float32) {
	n := float32(math.Sqrt(float64(DotProduct(v, v))))
	if n == 0 {
		return
	}
	inv := 1 / n
	for i := range v {
		v[i] *= inv
	}
}
