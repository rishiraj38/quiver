// Command quiver-bench builds an HNSW index over an fvecs dataset and
// measures recall, throughput and latency across a sweep of ef values.
// It runs on macOS and Linux.
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/rishiraj38/quiver/internal/hnsw"
	"github.com/rishiraj38/quiver/internal/vec"
)

type buildStats struct {
	Seconds        float64 `json:"seconds"`
	Threads        int     `json:"threads"`
	RSSBytes       int64   `json:"rss_bytes"`
	RSSBeforeBytes int64   `json:"rss_before_bytes"`
	PeakRSSBytes   int64   `json:"peak_rss_bytes"`
	HeapBytes      uint64  `json:"heap_bytes"`
	HeapBefore     uint64  `json:"heap_before_bytes"`
}

type serializeStats struct {
	WriteSeconds float64 `json:"write_seconds"`
	LoadSeconds  float64 `json:"load_seconds"`
	Bytes        int64   `json:"bytes"`
}

type searchRow struct {
	Ef         int     `json:"ef"`
	Recall     float64 `json:"recall"`
	QPSSingle  float64 `json:"qps_single"`
	QPSAll     float64 `json:"qps_all"`
	ThreadsAll int     `json:"threads_all"`
	P50Ms      float64 `json:"p50_ms"`
	P95Ms      float64 `json:"p95_ms"`
	P99Ms      float64 `json:"p99_ms"`
}

type systemInfo struct {
	CPU         string `json:"cpu"`
	LogicalCPUs int    `json:"logical_cpus"`
	OS          string `json:"os"`
	Arch        string `json:"arch"`
}

type result struct {
	Engine      string            `json:"engine"`
	Dataset     string            `json:"dataset"`
	Metric      string            `json:"metric"`
	N           int               `json:"n"`
	Dim         int               `json:"dim"`
	NQueries    int               `json:"n_queries"`
	K           int               `json:"k"`
	GroundTruth string            `json:"ground_truth"`
	Params      map[string]int    `json:"params"`
	Build       buildStats        `json:"build"`
	Serialize   *serializeStats   `json:"serialize"`
	Search      []searchRow       `json:"search"`
	Runs        int               `json:"runs"`
	System      systemInfo        `json:"system"`
	Versions    map[string]string `json:"versions"`
	Notes       []string          `json:"notes"`
	Timestamp   string            `json:"timestamp"`
}

// vectors is a row-major matrix of n rows.
type vectors struct {
	data []float32
	dim  int
	n    int
}

func (v vectors) row(i int) []float32 { return v.data[i*v.dim : (i+1)*v.dim : (i+1)*v.dim] }

func main() {
	log.SetFlags(0)
	var (
		dataDir = flag.String("data", "bench/datasets/sift-128-euclidean", "directory with base.fvecs, query.fvecs, groundtruth.ivecs")
		m       = flag.Int("m", 16, "HNSW M")
		efc     = flag.Int("efc", 200, "HNSW efConstruction")
		efList  = flag.String("efs", "10,20,40,80,120,200,400", "comma list of search ef values")
		k       = flag.Int("k", 10, "neighbours per query")
		threads = flag.Int("threads", runtime.NumCPU(), "build and all-cores search threads")
		limit   = flag.Int("limit", 0, "use only the first N base vectors (ground truth is then brute-forced)")
		runs    = flag.Int("runs", 3, "repeats per ef; the fastest is reported")
		metricF = flag.String("metric", "", "l2, cosine or dot (default: from meta.json, else l2)")
		out     = flag.String("out", "bench/results/quiver.json", "output JSON path")
	)
	flag.Parse()

	efs, err := parseInts(*efList)
	if err != nil || len(efs) == 0 {
		log.Fatalf("bad -efs %q", *efList)
	}
	if *threads < 1 || *k < 1 || *runs < 1 {
		log.Fatal("-threads, -k and -runs must be positive")
	}

	meta := readMeta(filepath.Join(*dataDir, "meta.json"))
	if *metricF == "" {
		*metricF = meta.Metric
	}
	metric, err := vec.ParseMetric(*metricF)
	if err != nil {
		log.Fatal(err)
	}
	dataset := meta.Dataset
	if dataset == "" {
		dataset = filepath.Base(filepath.Clean(*dataDir))
	}

	base, err := readFvecs(filepath.Join(*dataDir, "base.fvecs"), *limit)
	if err != nil {
		log.Fatal(err)
	}
	queries, err := readFvecs(filepath.Join(*dataDir, "query.fvecs"), 0)
	if err != nil {
		log.Fatal(err)
	}
	if base.dim != queries.dim {
		log.Fatalf("base dim %d != query dim %d", base.dim, queries.dim)
	}
	if metric == vec.Cosine {
		for i := 0; i < base.n; i++ {
			vec.Normalize(base.row(i))
		}
		for i := 0; i < queries.n; i++ {
			vec.Normalize(queries.row(i))
		}
	}
	log.Printf("%s: %d base, %d queries, dim %d, metric %s", dataset, base.n, queries.n, base.dim, metric)

	var truth [][]uint64
	truthKind := "shipped"
	if *limit > 0 {
		truthKind = "bruteforce"
		t0 := time.Now()
		truth = bruteForce(base, queries, *k, metric.Func(), *threads)
		log.Printf("brute-force ground truth in %.1fs", time.Since(t0).Seconds())
	} else {
		truth, err = readTruth(filepath.Join(*dataDir, "groundtruth.ivecs"), *k)
		if err != nil {
			log.Fatal(err)
		}
		if len(truth) != queries.n {
			log.Fatalf("ground truth has %d rows, want %d", len(truth), queries.n)
		}
	}

	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	heapBefore, rssBefore := ms.HeapAlloc, currentRSS()

	ix := hnsw.New(base.dim, hnsw.Config{M: *m, EfConstruction: *efc, Metric: metric})
	t0 := time.Now()
	if err := buildIndex(ix, base, *threads); err != nil {
		log.Fatal(err)
	}
	buildSec := time.Since(t0).Seconds()
	runtime.GC()
	runtime.ReadMemStats(&ms)
	build := buildStats{
		Seconds:        round(buildSec, 2),
		Threads:        *threads,
		RSSBytes:       currentRSS(),
		RSSBeforeBytes: rssBefore,
		PeakRSSBytes:   peakRSS(),
		HeapBytes:      ms.HeapAlloc,
		HeapBefore:     heapBefore,
	}
	log.Printf("build %.1fs, %d vectors, heap %d MB (index about %d MB)", buildSec, ix.Len(),
		ms.HeapAlloc>>20, (int64(ms.HeapAlloc)-int64(heapBefore))>>20)

	var rows []searchRow
	for _, ef := range efs {
		row := measure(ix, queries, truth, *k, ef, *threads, *runs)
		log.Printf("ef=%-4d recall=%.4f qps1=%.0f qpsN=%.0f p50=%.3fms p99=%.3fms",
			row.Ef, row.Recall, row.QPSSingle, row.QPSAll, row.P50Ms, row.P99Ms)
		rows = append(rows, row)
	}

	ser, err := measureSerialize(ix)
	if err != nil {
		log.Printf("serialise: %v", err)
	} else {
		log.Printf("write %.2fs, load %.2fs, %d MB", ser.WriteSeconds, ser.LoadSeconds, ser.Bytes>>20)
	}

	cfg := ix.Config()
	res := result{
		Engine:      "quiver",
		Dataset:     dataset,
		Metric:      metric.String(),
		N:           base.n,
		Dim:         base.dim,
		NQueries:    queries.n,
		K:           *k,
		GroundTruth: truthKind,
		Params:      map[string]int{"m": cfg.M, "ef_construction": cfg.EfConstruction},
		Build:       build,
		Serialize:   ser,
		Search:      rows,
		Runs:        *runs,
		System:      systemInfo{CPU: cpuModel(), LogicalCPUs: runtime.NumCPU(), OS: runtime.GOOS, Arch: runtime.GOARCH},
		Versions:    map[string]string{"go": runtime.Version()},
		Notes: []string{
			"In-process Go, no cgo, no SIMD assembly.",
			"qps_single: one goroutine, one Search call per query.",
			"qps_all: one Search call per query from -threads goroutines.",
			"recall is id overlap with the true top-k; ties in distance are not credited.",
		},
		Timestamp: time.Now().Format("2006-01-02T15:04:05-0700"),
	}
	if err := writeJSON(*out, res); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %s", *out)
}

func buildIndex(ix *hnsw.Index, base vectors, threads int) error {
	var (
		next     atomic.Int64
		failed   atomic.Bool
		firstErr error
		once     sync.Once
		wg       sync.WaitGroup
	)
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !failed.Load() {
				i := int(next.Add(1)) - 1
				if i >= base.n {
					return
				}
				if err := ix.Add(uint64(i), base.row(i)); err != nil {
					once.Do(func() { firstErr = fmt.Errorf("add %d: %w", i, err) })
					failed.Store(true)
					return
				}
			}
		}()
	}
	wg.Wait()
	return firstErr
}

// measure runs the query set at one ef: runs single-goroutine passes for
// recall, latency and single-thread QPS, then runs passes across all threads.
func measure(ix *hnsw.Index, queries vectors, truth [][]uint64, k, ef, threads, runs int) searchRow {
	n := queries.n
	found := make([][]hnsw.Result, n)
	lat := make([]time.Duration, n)
	var bestLat []time.Duration
	best := time.Duration(math.MaxInt64)
	for r := 0; r < runs; r++ {
		start := time.Now()
		for i := 0; i < n; i++ {
			t0 := time.Now()
			found[i] = ix.Search(queries.row(i), k, ef, nil)
			lat[i] = time.Since(t0)
		}
		if total := time.Since(start); total < best {
			best = total
			bestLat = append(bestLat[:0], lat...)
		}
	}

	hits := 0
	for i, res := range found {
		want := truth[i]
		for _, r := range res {
			for _, id := range want {
				if r.ID == id {
					hits++
					break
				}
			}
		}
	}

	bestAll := time.Duration(math.MaxInt64)
	for r := 0; r < runs; r++ {
		var next atomic.Int64
		var wg sync.WaitGroup
		start := time.Now()
		for w := 0; w < threads; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := int(next.Add(1)) - 1
					if i >= n {
						return
					}
					ix.Search(queries.row(i), k, ef, nil)
				}
			}()
		}
		wg.Wait()
		if total := time.Since(start); total < bestAll {
			bestAll = total
		}
	}

	sort.Slice(bestLat, func(a, b int) bool { return bestLat[a] < bestLat[b] })
	return searchRow{
		Ef:         ef,
		Recall:     round(float64(hits)/float64(n*k), 5),
		QPSSingle:  round(float64(n)/best.Seconds(), 1),
		QPSAll:     round(float64(n)/bestAll.Seconds(), 1),
		ThreadsAll: threads,
		P50Ms:      round(percentile(bestLat, 0.50), 4),
		P95Ms:      round(percentile(bestLat, 0.95), 4),
		P99Ms:      round(percentile(bestLat, 0.99), 4),
	}
}

func measureSerialize(ix *hnsw.Index) (*serializeStats, error) {
	f, err := os.CreateTemp("", "quiver-bench-*.idx")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)
	t0 := time.Now()
	if _, err := ix.WriteTo(w); err != nil {
		return nil, err
	}
	if err := w.Flush(); err != nil {
		return nil, err
	}
	writeSec := time.Since(t0).Seconds()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}
	t0 = time.Now()
	loaded, err := hnsw.Load(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return nil, err
	}
	loadSec := time.Since(t0).Seconds()
	if loaded.Len() != ix.Len() {
		return nil, fmt.Errorf("loaded index has %d vectors, want %d", loaded.Len(), ix.Len())
	}
	return &serializeStats{WriteSeconds: round(writeSec, 3), LoadSeconds: round(loadSec, 3), Bytes: st.Size()}, nil
}

// bruteForce returns the exact k nearest base ids for every query.
func bruteForce(base, queries vectors, k int, dist vec.DistFunc, threads int) [][]uint64 {
	if k > base.n {
		k = base.n
	}
	out := make([][]uint64, queries.n)
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ids := make([]uint64, 0, k+1)
			ds := make([]float32, 0, k+1)
			for {
				qi := int(next.Add(1)) - 1
				if qi >= queries.n {
					return
				}
				q := queries.row(qi)
				ids, ds = ids[:0], ds[:0]
				for i := 0; i < base.n; i++ {
					d := dist(q, base.row(i))
					if len(ds) == k && d >= ds[k-1] {
						continue
					}
					pos := sort.Search(len(ds), func(j int) bool { return ds[j] > d })
					if len(ds) < k {
						ids, ds = append(ids, 0), append(ds, 0)
					}
					copy(ids[pos+1:], ids[pos:])
					copy(ds[pos+1:], ds[pos:])
					ids[pos], ds[pos] = uint64(i), d
				}
				out[qi] = append([]uint64(nil), ids...)
			}
		}()
	}
	wg.Wait()
	return out
}

// readFvecs reads up to limit rows (all when limit <= 0). Each row is a
// little-endian int32 dimension followed by that many float32 values.
func readFvecs(path string, limit int) (vectors, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return vectors{}, err
	}
	if len(raw) < 4 {
		return vectors{}, fmt.Errorf("%s: empty", path)
	}
	dim := int(int32(binary.LittleEndian.Uint32(raw)))
	rowBytes := 4 + 4*dim
	if dim <= 0 || len(raw)%rowBytes != 0 {
		return vectors{}, fmt.Errorf("%s: size %d does not fit dimension %d", path, len(raw), dim)
	}
	n := len(raw) / rowBytes
	if limit > 0 && limit < n {
		n = limit
	}
	data := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		row := raw[i*rowBytes : (i+1)*rowBytes]
		if int(int32(binary.LittleEndian.Uint32(row))) != dim {
			return vectors{}, fmt.Errorf("%s: row %d has a different dimension", path, i)
		}
		dst := data[i*dim : (i+1)*dim]
		for j := range dst {
			dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(row[4+4*j:]))
		}
	}
	return vectors{data: data, dim: dim, n: n}, nil
}

// readTruth reads an ivecs file and keeps the first k ids of each row.
func readTruth(path string, k int) ([][]uint64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 4 {
		return nil, fmt.Errorf("%s: empty", path)
	}
	dim := int(int32(binary.LittleEndian.Uint32(raw)))
	rowBytes := 4 + 4*dim
	if dim <= 0 || len(raw)%rowBytes != 0 {
		return nil, fmt.Errorf("%s: size %d does not fit dimension %d", path, len(raw), dim)
	}
	if k > dim {
		return nil, fmt.Errorf("%s: holds %d neighbours per query, need %d", path, dim, k)
	}
	out := make([][]uint64, len(raw)/rowBytes)
	for i := range out {
		row := raw[i*rowBytes+4:]
		ids := make([]uint64, k)
		for j := range ids {
			ids[j] = uint64(binary.LittleEndian.Uint32(row[4*j:]))
		}
		out[i] = ids
	}
	return out, nil
}

type metaFile struct {
	Dataset string `json:"dataset"`
	Metric  string `json:"metric"`
}

// readMeta returns the zero value when meta.json is missing or unreadable.
func readMeta(path string) metaFile {
	var m metaFile
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, &m)
	}
	return m
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("bad value %q", f)
		}
		out = append(out, n)
	}
	return out, nil
}

// percentile expects sorted input and returns milliseconds.
func percentile(sorted []time.Duration, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(math.Ceil(p*float64(len(sorted)))) - 1
	if i < 0 {
		i = 0
	}
	return float64(sorted[i]) / float64(time.Millisecond)
}

func round(x float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(x*p) / p
}

// currentRSS asks ps for the resident set size; 0 if that fails.
func currentRSS() int64 {
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return 0
	}
	kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0
	}
	return kb * 1024
}

// peakRSS reports ru_maxrss, which is bytes on macOS and kilobytes on Linux.
func peakRSS() int64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	if runtime.GOOS == "darwin" {
		return int64(ru.Maxrss)
	}
	return int64(ru.Maxrss) * 1024
}

func cpuModel() string {
	if runtime.GOOS == "darwin" {
		if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			return strings.TrimSpace(string(out))
		}
		return "unknown"
	}
	if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "model name") {
				if _, v, ok := strings.Cut(line, ":"); ok {
					return strings.TrimSpace(v)
				}
			}
		}
	}
	return "unknown"
}
