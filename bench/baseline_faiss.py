"""FAISS IndexHNSWFlat baseline."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss

import common


def main():
    args = common.base_parser("faiss").parse_args()
    base, queries, truth, meta = common.load_dataset(args)
    n, dim = base.shape
    rss_before = common.rss_bytes()

    metric = faiss.METRIC_L2 if meta["metric"] == "l2" else faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexHNSWFlat(dim, args.m, metric)
    index.hnsw.efConstruction = args.ef_construction

    faiss.omp_set_num_threads(args.threads)
    t0 = time.perf_counter()
    index.add(base)
    build_s = time.perf_counter() - t0
    build = {
        "seconds": round(build_s, 2),
        "threads": args.threads,
        "rss_bytes": common.rss_bytes(),
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": common.peak_rss_bytes(),
    }
    print(f"build {build_s:.1f}s", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.faiss")
        t0 = time.perf_counter()
        faiss.write_index(index, path)
        write_s = time.perf_counter() - t0
        size = os.path.getsize(path)
        t0 = time.perf_counter()
        faiss.read_index(path)
        load_s = time.perf_counter() - t0
    serialize = {"write_seconds": round(write_s, 3), "load_seconds": round(load_s, 3), "bytes": size}

    def set_ef(ef):
        index.hnsw.efSearch = ef

    def search_one(q):
        return index.search(q.reshape(1, -1), args.k)[1][0]

    def all_cores(qs):
        t0 = time.perf_counter()
        index.search(qs, args.k)
        return time.perf_counter() - t0

    search = common.sweep(
        args,
        queries,
        truth,
        set_ef,
        search_one,
        all_cores,
        set_single=lambda: faiss.omp_set_num_threads(1),
        set_all=lambda: faiss.omp_set_num_threads(args.threads),
    )

    notes = [
        "IndexHNSWFlat, in-process, called from Python through SWIG.",
        "qps_single: omp threads = 1, one query per search() call.",
        "qps_all: all queries in one search() call, parallelised by FAISS over OpenMP threads.",
        "recall is id overlap with the true top-k; ties in distance are not credited.",
    ]
    common.write_result(
        args, "faiss", meta, n, build, search, {"faiss": faiss.__version__}, notes, serialize
    )


if __name__ == "__main__":
    main()
