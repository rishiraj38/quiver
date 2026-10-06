package pq

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/rishiraj38/quiver/internal/vec"
)

// clustered draws n vectors from a mixture of Gaussians. Uniform noise has
// no structure for a codebook to learn, so it would say nothing about
// whether training works.
func clustered(n, dim, clusters int, sigma float32, seed uint64) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, 0))
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = float32(rng.NormFloat64())
		}
	}
	flat := make([]float32, n*dim)
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.IntN(clusters)]
		v := flat[i*dim : (i+1)*dim : (i+1)*dim]
		for j := range v {
			v[j] = c[j] + sigma*float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// lowRank is a mixture of Gaussians whose within-cluster covariance has rank
// `rank` plus a little isotropic noise, the shape real embeddings have:
// correlated dimensions and an intrinsic dimension well below dim. With
// isotropic clusters every member is nearly equidistant from a query in 64-d,
// and no 8-byte code can rank them.
func lowRank(n, dim, clusters, rank int, noise float32, seed uint64) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, 1))
	normal := func(k int) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = normal(dim)
	}
	basis := make([][]float32, rank)
	for i := range basis {
		basis[i] = normal(dim)
	}
	scale := 1 / float32(math.Sqrt(float64(rank)))

	flat := make([]float32, n*dim)
	out := make([][]float32, n)
	for i := range out {
		v := flat[i*dim : (i+1)*dim : (i+1)*dim]
		copy(v, centers[rng.IntN(clusters)])
		for _, b := range basis {
			z := scale * float32(rng.NormFloat64())
			for j := range v {
				v[j] += z * b[j]
			}
		}
		for j := range v {
			v[j] += noise * float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

func mustTrain(tb testing.TB, sample [][]float32, m, iters int, seed int64) *Quantizer {
	tb.Helper()
	q, err := Train(sample, m, iters, seed)
	if err != nil {
		tb.Fatalf("Train: %v", err)
	}
	return q
}

func reconstructionError(q *Quantizer, data [][]float32) float64 {
	var total float64
	var code []byte
	var rec []float32
	for _, v := range data {
		code = q.Encode(code[:0], v)
		rec = q.Decode(rec[:0], code)
		total += float64(vec.L2Squared(v, rec))
	}
	return total / float64(len(data))
}

func TestTrainErrors(t *testing.T) {
	good := clustered(300, 16, 8, 0.2, 1)
	ragged := append([][]float32{}, good...)
	ragged[17] = ragged[17][:12]

	cases := []struct {
		name   string
		sample [][]float32
		m      int
	}{
		{"dim not divisible by m", good, 5},
		{"m zero", good, 0},
		{"m negative", good, -4},
		{"m larger than dim", good, 32},
		{"too few samples", good[:255], 4},
		{"no samples", nil, 4},
		{"ragged", ragged, 4},
		{"zero dimension", make([][]float32, 300), 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if q, err := Train(tc.sample, tc.m, 5, 1); err == nil {
				t.Fatalf("got quantizer %+v, want error", q)
			}
		})
	}
}

func TestShape(t *testing.T) {
	q := mustTrain(t, clustered(300, 16, 8, 0.2, 1), 4, 5, 1)
	if q.Dim() != 16 || q.M() != 4 {
		t.Fatalf("Dim, M = %d, %d, want 16, 4", q.Dim(), q.M())
	}
	if got, want := len(q.codebooks), 4*256*4; got != want {
		t.Fatalf("codebook holds %d floats, want %d", got, want)
	}
}

func TestReconstructionErrorDropsWithM(t *testing.T) {
	data := clustered(4000, 64, 32, 0.3, 2)
	prev := math.Inf(1)
	for _, m := range []int{4, 8, 16, 32} {
		q := mustTrain(t, data, m, 10, 7)
		mse := reconstructionError(q, data)
		t.Logf("m=%-2d  mean squared reconstruction error %.4f", m, mse)
		if mse >= prev {
			t.Errorf("m=%d: error %.4f did not drop below %.4f", m, mse, prev)
		}
		prev = mse
	}
}

// Lloyd's algorithm never increases the objective, so more iterations must
// not reconstruct worse.
func TestMoreIterationsDoNotHurt(t *testing.T) {
	data := clustered(3000, 32, 16, 0.3, 3)
	one := reconstructionError(mustTrain(t, data, 8, 1, 5), data)
	many := reconstructionError(mustTrain(t, data, 8, 15, 5), data)
	if many > one {
		t.Fatalf("15 iterations gave error %.5f, worse than 1 iteration's %.5f", many, one)
	}
}

func TestEncodePicksNearestCentroid(t *testing.T) {
	data := clustered(1000, 16, 8, 0.3, 4)
	q := mustTrain(t, data, 4, 5, 1)
	for _, v := range data[:50] {
		code := q.Encode(nil, v)
		for s := 0; s < q.m; s++ {
			sub := v[s*q.dsub : (s+1)*q.dsub]
			cb := q.codebook(s)
			chosen := vec.L2Squared(sub, cb[int(code[s])*q.dsub:(int(code[s])+1)*q.dsub])
			for c := 0; c < ksub; c++ {
				if d := vec.L2Squared(sub, cb[c*q.dsub:(c+1)*q.dsub]); d < chosen {
					t.Fatalf("sub-space %d: centroid %d at %g beats chosen %d at %g", s, c, d, code[s], chosen)
				}
			}
		}
	}
}

func TestADCMatchesDecodedDistance(t *testing.T) {
	data := clustered(2000, 64, 16, 0.3, 5)
	// m=10 would not divide 64; 16 and 4 exercise the unrolled loop, and
	// dim 60 with m=6 exercises its scalar tail.
	for _, shape := range []struct{ dim, m int }{{64, 16}, {64, 4}, {60, 6}, {64, 1}} {
		sample := data
		if shape.dim != 64 {
			sample = clustered(2000, shape.dim, 16, 0.3, 5)
		}
		q := mustTrain(t, sample, shape.m, 5, 1)
		queries := clustered(20, shape.dim, 16, 0.3, 99)
		var rec []float32
		for _, query := range queries {
			table := q.NewTable(query)
			for _, v := range sample[:200] {
				code := q.Encode(nil, v)
				rec = q.Decode(rec[:0], code)
				want := vec.L2Squared(query, rec)
				got := table.Distance(code)
				if diff := math.Abs(float64(got - want)); diff > 1e-4*math.Max(1, float64(want)) {
					t.Fatalf("dim=%d m=%d: ADC %g, exact distance to decoded vector %g", shape.dim, shape.m, got, want)
				}
			}
		}
	}
}

func TestDeterministic(t *testing.T) {
	data := clustered(2000, 32, 16, 0.3, 6)
	marshal := func(seed int64) []byte {
		b, err := mustTrain(t, data, 8, 8, seed).MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	a, b := marshal(42), marshal(42)
	if !bytes.Equal(a, b) {
		t.Fatal("two runs with the same seed produced different codebooks")
	}
	if bytes.Equal(a, marshal(43)) {
		t.Fatal("different seeds produced identical codebooks")
	}
}

// With far fewer distinct points than centroids, most clusters start empty.
// Training must terminate and still reconstruct the points it saw exactly.
func TestDegenerateSample(t *testing.T) {
	distinct := clustered(10, 8, 10, 1, 7)
	sample := make([][]float32, 600)
	for i := range sample {
		sample[i] = distinct[i%len(distinct)]
	}
	q := mustTrain(t, sample, 2, 10, 1)
	for _, f := range q.codebooks {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatal("codebook contains a non-finite value")
		}
	}
	if mse := reconstructionError(q, distinct); mse > 1e-9 {
		t.Fatalf("reconstruction error %g on a sample with 10 distinct points, want 0", mse)
	}
}

func TestFillEmptyReseedsFromFarthestPoint(t *testing.T) {
	km := newKMeans(ksub+1, 1)
	for i := range km.assign {
		km.assign[i] = int32(i)
		km.counts[i%ksub]++
	}
	// Points 0 and 256 share cluster 0, leaving cluster 255 empty only
	// after we move its sole member away.
	km.assign[ksub] = 0
	km.assign[255] = 1
	km.counts[255], km.counts[1] = 0, 2
	km.dist[1], km.dist[255], km.dist[0] = 0.5, 3, 9
	km.dist[ksub] = 2

	if !km.fillEmpty() {
		t.Fatal("fillEmpty reported nothing to do")
	}
	// Point 0 is farthest (9) and its cluster has a second member.
	if km.assign[0] != 255 {
		t.Fatalf("point 0 assigned to %d, want the empty cluster 255", km.assign[0])
	}
	for c, n := range km.counts {
		if n == 0 {
			t.Fatalf("cluster %d is still empty", c)
		}
	}
}

func TestAppendSemantics(t *testing.T) {
	data := clustered(500, 16, 8, 0.3, 8)
	q := mustTrain(t, data, 4, 5, 1)

	prefix := []byte{0xAA, 0xBB}
	codes := q.Encode(prefix, data[0])
	if len(codes) != 2+q.M() || codes[0] != 0xAA || codes[1] != 0xBB {
		t.Fatalf("Encode did not append: % x", codes)
	}
	recs := q.Decode([]float32{1.5}, codes[2:])
	if len(recs) != 1+q.Dim() || recs[0] != 1.5 {
		t.Fatalf("Decode did not append: len %d, first %g", len(recs), recs[0])
	}

	code := make([]byte, 0, q.M())
	rec := make([]float32, 0, q.Dim())
	table := q.NewTable(data[1])
	var sink float32
	allocs := testing.AllocsPerRun(100, func() {
		code = q.Encode(code[:0], data[0])
		rec = q.Decode(rec[:0], code)
		sink += table.Distance(code)
	})
	if allocs != 0 {
		t.Fatalf("Encode+Decode+Distance allocated %.0f times with capacity available", allocs)
	}
	_ = sink
}

func TestMarshalRoundTrip(t *testing.T) {
	data := clustered(1000, 32, 16, 0.3, 9)
	q := mustTrain(t, data, 8, 5, 1)
	blob, err := q.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if want := headerSize + 4*ksub*32 + trailerSize; len(blob) != want {
		t.Fatalf("marshalled size %d, want %d", len(blob), want)
	}

	var got Quantizer
	if err := got.UnmarshalBinary(blob); err != nil {
		t.Fatal(err)
	}
	if got.Dim() != q.Dim() || got.M() != q.M() {
		t.Fatalf("shape %dx%d, want %dx%d", got.Dim(), got.M(), q.Dim(), q.M())
	}
	for _, v := range data[:100] {
		if a, b := q.Encode(nil, v), got.Encode(nil, v); !bytes.Equal(a, b) {
			t.Fatalf("codes differ after round trip: % x vs % x", a, b)
		}
	}
	again, err := got.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(blob, again) {
		t.Fatal("re-marshalling a decoded quantizer changed the bytes")
	}
}

// reseal recomputes the trailing checksum, to build inputs that are
// internally consistent but describe an invalid quantizer.
func reseal(b []byte) []byte {
	body := b[:len(b)-trailerSize]
	return binary.LittleEndian.AppendUint32(body[:len(body):len(body)], crc32.Checksum(body, crcTable))
}

func TestUnmarshalRejectsBadInput(t *testing.T) {
	q := mustTrain(t, clustered(500, 16, 8, 0.3, 10), 4, 3, 1)
	blob, err := q.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	mutate := func(f func(b []byte) []byte) []byte {
		return f(bytes.Clone(blob))
	}
	setU32 := func(off int, v uint32) func([]byte) []byte {
		return func(b []byte) []byte {
			binary.LittleEndian.PutUint32(b[off:], v)
			return reseal(b)
		}
	}

	cases := map[string][]byte{
		"nil":                nil,
		"empty":              {},
		"header only":        blob[:headerSize],
		"truncated":          blob[:len(blob)-9],
		"trailing garbage":   append(bytes.Clone(blob), 0, 0, 0, 0),
		"bad magic":          mutate(func(b []byte) []byte { b[0] = 'X'; return reseal(b) }),
		"future version":     mutate(setU32(4, formatVersion+1)),
		"dim zero":           mutate(setU32(8, 0)),
		"dim huge":           mutate(setU32(8, math.MaxUint32)),
		"dim mismatch":       mutate(setU32(8, 32)),
		"m zero":             mutate(setU32(12, 0)),
		"m not dividing":     mutate(setU32(12, 5)),
		"m huge":             mutate(setU32(12, math.MaxUint32)),
		"wrong ksub":         mutate(setU32(16, 16)),
		"flipped body bit":   mutate(func(b []byte) []byte { b[headerSize+100] ^= 0x01; return b }),
		"flipped crc bit":    mutate(func(b []byte) []byte { b[len(b)-1] ^= 0x80; return b }),
		"flipped dim bit":    mutate(func(b []byte) []byte { b[8] ^= 0x01; return b }),
		"all zero":           make([]byte, len(blob)),
		"shorter than a u32": {'Q', 'V'},
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			got := Quantizer{dim: 7, m: 7}
			if err := got.UnmarshalBinary(data); err == nil {
				t.Fatal("accepted bad input")
			}
			if got.dim != 7 || got.m != 7 || got.codebooks != nil {
				t.Fatal("failed UnmarshalBinary modified the receiver")
			}
		})
	}

	if _, err := new(Quantizer).MarshalBinary(); err == nil {
		t.Fatal("marshalled an untrained quantizer")
	}
}

// Every single-bit flip in the header and a sample of the body must be
// caught, and none may panic.
func TestUnmarshalBitFlips(t *testing.T) {
	q := mustTrain(t, clustered(300, 8, 8, 0.3, 11), 2, 2, 1)
	blob, _ := q.MarshalBinary()
	rng := rand.New(rand.NewPCG(1, 2))
	check := func(pos int) {
		b := bytes.Clone(blob)
		b[pos/8] ^= 1 << (pos % 8)
		if err := new(Quantizer).UnmarshalBinary(b); err == nil {
			t.Fatalf("bit %d flipped and input still accepted", pos)
		}
	}
	for pos := 0; pos < headerSize*8; pos++ {
		check(pos)
	}
	for range 2000 {
		check(rng.IntN(len(blob) * 8))
	}
}

func FuzzUnmarshalBinary(f *testing.F) {
	q := mustTrain(f, clustered(300, 8, 8, 0.3, 11), 2, 2, 1)
	blob, _ := q.MarshalBinary()
	f.Add(blob)
	f.Add(blob[:headerSize])
	f.Add([]byte(magic))
	f.Fuzz(func(t *testing.T, data []byte) {
		var got Quantizer
		if err := got.UnmarshalBinary(data); err != nil {
			return
		}
		// Anything accepted must be usable without panicking.
		code := got.Encode(nil, make([]float32, got.Dim()))
		got.Decode(nil, code)
		got.NewTable(make([]float32, got.Dim())).Distance(code)
	})
}

func BenchmarkTrain(b *testing.B) {
	data := clustered(10000, 64, 64, 0.3, 1)
	b.ResetTimer()
	for b.Loop() {
		mustTrain(b, data, 8, 10, 1)
	}
}

func BenchmarkEncode(b *testing.B) {
	data := clustered(4000, 64, 64, 0.3, 1)
	q := mustTrain(b, data, 8, 5, 1)
	code := make([]byte, 0, q.M())
	b.ReportAllocs()
	b.ResetTimer()
	i := 0
	for b.Loop() {
		code = q.Encode(code[:0], data[i%len(data)])
		i++
	}
}

func BenchmarkNewTable(b *testing.B) {
	data := clustered(4000, 64, 64, 0.3, 1)
	q := mustTrain(b, data, 8, 5, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		q.NewTable(data[0])
	}
}

func BenchmarkTableDistance(b *testing.B) {
	data := clustered(4000, 64, 64, 0.3, 1)
	for _, m := range []int{8, 16, 32} {
		b.Run(fmt.Sprintf("m=%d", m), func(b *testing.B) {
			q := mustTrain(b, data, m, 5, 1)
			// A realistic scan walks many codes, not one hot code.
			var codes []byte
			for _, v := range data {
				codes = q.Encode(codes, v)
			}
			table := q.NewTable(data[0])
			var sink float32
			b.ReportAllocs()
			b.ResetTimer()
			i := 0
			for b.Loop() {
				sink += table.Distance(codes[i*m : (i+1)*m])
				if i++; i == len(data) {
					i = 0
				}
			}
			_ = sink
		})
	}
}
