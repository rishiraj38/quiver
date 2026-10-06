# Benchmarks

Recall vs throughput for quiver's HNSW against FAISS, hnswlib and Qdrant on the
ann-benchmarks `sift-128-euclidean` set (1,000,000 base vectors, 10,000 queries,
128 dimensions, L2). Every engine uses M=16, efConstruction=200 and the same
search sweep: ef = 10, 20, 40, 80, 120, 200, 400.

## Reproduce

From the repository root. Run the steps one at a time; each holds the full
dataset in memory.

```sh
uv venv bench/.venv
uv pip install --python bench/.venv/bin/python -r bench/requirements.txt

bench/.venv/bin/python bench/prepare.py            # ~500 MB download, converts to fvecs/ivecs
bench/.venv/bin/python bench/baseline_faiss.py
bench/.venv/bin/python bench/baseline_hnswlib.py
bench/.venv/bin/python bench/baseline_qdrant.py    # needs Docker running
go run ./cmd/quiver-bench
bench/.venv/bin/python bench/plot.py
```

Each step writes `bench/results/<engine>.json`. `plot.py` draws whichever of
those exist to `bench/results/recall_qps.png` and `.svg` and prints a markdown
table.

Useful flags, shared by all four runners: `--limit N` (`-limit N` for
quiver-bench) uses the first N base vectors and brute-forces its own ground
truth; `--efs`, `--k`, `--threads`, `--runs`, `--out`. Another ann-benchmarks
set works the same way: `prepare.py --dataset glove-100-angular`, then pass
`--dataset glove-100-angular` (`-data bench/datasets/glove-100-angular` for
quiver-bench).

`baseline_qdrant.py` starts a container named `quiver-bench-qdrant` with
`--memory 5g`, storage in `bench/qdrant_storage/`, and removes both when it
finishes. It touches no other container.

## What is measured

- `recall`: mean overlap between the returned ids and the true top k, k=10.
  Ties in distance are not credited, the same for every engine.
- `qps_single`: one thread, one query per call, 10,000 queries in order.
  FAISS runs with `omp_set_num_threads(1)`, hnswlib with `num_threads=1`.
- `qps_all`: all cores. quiver issues one `Search` per query from N goroutines;
  FAISS and hnswlib get the whole query matrix in one call and fan out
  internally; Qdrant gets one request per query from a client thread pool.
- `p50/p95/p99_ms`: per-query latency from the single-thread pass.
- Each ef is run `--runs` times (default 3) and the fastest pass is reported.
- `build.seconds`: wall time to insert every vector using all cores.

## Caveats

- **Qdrant is measured over the network.** It runs in Docker and is queried
  through its gRPC client, so each query pays serialisation, a loopback round
  trip and Python client overhead that the in-process engines do not. On macOS
  the container also runs inside Docker's Linux VM with its own CPU and memory
  limits. Its recall is directly comparable; its QPS is not a like-for-like
  number. The JSON carries this in `notes`.
- **Thread pinning.** Single-thread numbers pin the library's own thread pool
  to one thread. Nothing pins the process to a core, and Apple Silicon mixes
  performance and efficiency cores, so the scheduler decides where the thread
  runs. All-cores numbers use every logical core, efficiency cores included.
- **Call overhead.** FAISS and hnswlib are called from Python once per query,
  which costs a few microseconds per call. That matters at low ef, where a
  query takes tens of microseconds, and is noise at high ef.
- **SIMD.** FAISS ships NEON kernels for arm64. hnswlib's hand-written kernels
  are SSE/AVX only, so on arm64 it runs its scalar loop as vectorised by the
  compiler. quiver is pure Go with no assembly.
- **Hardware.** Numbers in `results/` come from one machine (CPU model and
  core count are recorded in each JSON). Compare engines within a run, not
  against numbers from other hardware.
- **Same parameters are not the same graph.** The engines differ in neighbour
  selection heuristics and level assignment, so at equal ef they land on
  slightly different recall. Read the curve, not a single ef.
