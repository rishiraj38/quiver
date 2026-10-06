"""hnswlib baseline."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib.metadata import version

import hnswlib
import numpy as np

import common


def main():
    args = common.base_parser("hnswlib").parse_args()
    base, queries, truth, meta = common.load_dataset(args)
    n, dim = base.shape
    rss_before = common.rss_bytes()

    space = "l2" if meta["metric"] == "l2" else "ip"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, M=args.m, ef_construction=args.ef_construction, random_seed=100)

    t0 = time.perf_counter()
    index.add_items(base, np.arange(n), num_threads=args.threads)
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
        path = os.path.join(tmp, "index.bin")
        t0 = time.perf_counter()
        index.save_index(path)
        write_s = time.perf_counter() - t0
        size = os.path.getsize(path)
        other = hnswlib.Index(space=space, dim=dim)
        t0 = time.perf_counter()
        other.load_index(path, max_elements=n)
        load_s = time.perf_counter() - t0
        del other
    serialize = {"write_seconds": round(write_s, 3), "load_seconds": round(load_s, 3), "bytes": size}

    def search_one(q):
        return index.knn_query(q, k=args.k, num_threads=1)[0][0]

    def all_cores(qs):
        t0 = time.perf_counter()
        index.knn_query(qs, k=args.k, num_threads=args.threads)
        return time.perf_counter() - t0

    search = common.sweep(args, queries, truth, index.set_ef, search_one, all_cores)

    notes = [
        "In-process, called from Python through pybind11.",
        "qps_single: num_threads = 1, one query per knn_query() call.",
        "qps_all: all queries in one knn_query() call, parallelised by hnswlib's thread pool.",
        "recall is id overlap with the true top-k; ties in distance are not credited.",
    ]
    common.write_result(
        args, "hnswlib", meta, n, build, search, {"hnswlib": version("hnswlib")}, notes, serialize
    )


if __name__ == "__main__":
    main()
