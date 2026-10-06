package vec

import (
	"math"
	"math/rand/v2"
	"testing"
)

func naiveL2(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

func naiveDot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// The unrolled kernels have a 4-wide body and a scalar tail; lengths around
// the boundary exercise both.
func TestKernelsMatchNaive(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 19, 20, 23, 31, 33, 100, 127, 128, 129, 960, 1536} {
		a, b := make([]float32, n), make([]float32, n)
		for i := range a {
			a[i], b[i] = r.Float32()*2-1, r.Float32()*2-1
		}
		if got, want := float64(L2Squared(a, b)), naiveL2(a, b); math.Abs(got-want) > 1e-4*(1+want) {
			t.Errorf("L2Squared n=%d: got %g, want %g", n, got, want)
		}
		if got, want := float64(DotProduct(a, b)), naiveDot(a, b); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
			t.Errorf("DotProduct n=%d: got %g, want %g", n, got, want)
		}
		// On arm64 the exported kernels are assembly; keep the portable
		// ones honest too.
		if got, want := float64(l2SquaredGeneric(a, b)), naiveL2(a, b); math.Abs(got-want) > 1e-4*(1+want) {
			t.Errorf("l2SquaredGeneric n=%d: got %g, want %g", n, got, want)
		}
		if got, want := float64(dotGeneric(a, b)), naiveDot(a, b); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
			t.Errorf("dotGeneric n=%d: got %g, want %g", n, got, want)
		}
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4, 0}
	Normalize(v)
	if n := DotProduct(v, v); math.Abs(float64(n)-1) > 1e-6 {
		t.Errorf("norm after Normalize = %g", n)
	}
	zero := []float32{0, 0}
	Normalize(zero)
	if zero[0] != 0 || zero[1] != 0 {
		t.Errorf("zero vector changed: %v", zero)
	}
}

func TestMetrics(t *testing.T) {
	a, b := []float32{1, 0}, []float32{0, 1}
	if d := L2.Func()(a, b); d != 2 {
		t.Errorf("l2 = %g, want 2", d)
	}
	if d := Cosine.Func()(a, a); d != 0 {
		t.Errorf("cosine self = %g, want 0", d)
	}
	if d := Cosine.Func()(a, b); d != 1 {
		t.Errorf("cosine orthogonal = %g, want 1", d)
	}
	if d := Dot.Func()(a, a); d != -1 {
		t.Errorf("dot self = %g, want -1", d)
	}
	for _, m := range []Metric{L2, Cosine, Dot} {
		got, err := ParseMetric(m.String())
		if err != nil || got != m {
			t.Errorf("round trip %v: %v %v", m, got, err)
		}
	}
	if _, err := ParseMetric("hamming"); err == nil {
		t.Error("unknown metric accepted")
	}
}

func BenchmarkL2Squared128(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	x, y := make([]float32, 128), make([]float32, 128)
	for i := range x {
		x[i], y[i] = r.Float32(), r.Float32()
	}
	var sink float32
	for b.Loop() {
		sink += L2Squared(x, y)
	}
	_ = sink
}

func BenchmarkL2SquaredGeneric128(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	x, y := make([]float32, 128), make([]float32, 128)
	for i := range x {
		x[i], y[i] = r.Float32(), r.Float32()
	}
	var sink float32
	for b.Loop() {
		sink += l2SquaredGeneric(x, y)
	}
	_ = sink
}

func BenchmarkDotProduct128(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	x, y := make([]float32, 128), make([]float32, 128)
	for i := range x {
		x[i], y[i] = r.Float32(), r.Float32()
	}
	var sink float32
	for b.Loop() {
		sink += DotProduct(x, y)
	}
	_ = sink
}
