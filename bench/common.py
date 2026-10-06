"""Shared helpers for the Python baselines: dataset loading, timing, JSON output."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EFS = "10,20,40,80,120,200,400"


def base_parser(engine):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--efs", default=DEFAULT_EFS)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="build and all-cores threads")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N base vectors")
    ap.add_argument("--runs", type=int, default=3, help="repeats per ef; the fastest is reported")
    ap.add_argument("--out", default=os.path.join(HERE, "results", engine + ".json"))
    return ap


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype="<i4")
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if not (rows[:, 0] == d).all():
        raise ValueError(f"{path}: inconsistent dimension header")
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def exact_neighbours(base, queries, k, metric):
    """Brute-force top-k ids, used when --limit makes the shipped truth invalid."""
    out = np.empty((len(queries), k), dtype=np.int32)
    base_t = base.T.astype(np.float64)
    base_sq = (base_t**2).sum(axis=0)
    for i in range(0, len(queries), 500):
        q = queries[i : i + 500].astype(np.float64)
        ip = q @ base_t
        d = -ip if metric != "l2" else base_sq[None, :] - 2 * ip
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1)
        out[i : i + 500] = np.take_along_axis(part, order, axis=1)
    return out


def load_dataset(args):
    """Returns (base, queries, truth[:, :k], meta). Cosine data is normalised."""
    root = os.path.join(args.data_dir, args.dataset)
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)
    base = read_vecs(os.path.join(root, "base.fvecs"), "<f4")
    queries = read_vecs(os.path.join(root, "query.fvecs"), "<f4")
    if meta["metric"] == "cosine":
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        base, queries = base.astype("<f4"), queries.astype("<f4")
    if args.limit and args.limit < len(base):
        base = np.ascontiguousarray(base[: args.limit])
        truth = exact_neighbours(base, queries, args.k, meta["metric"])
        meta["ground_truth"] = "bruteforce"
    else:
        truth = read_vecs(os.path.join(root, "groundtruth.ivecs"), "<i4")[:, : args.k]
        meta["ground_truth"] = "shipped"
    return base, queries, truth, meta


def recall_at_k(found, truth, k):
    """found: list of id sequences, one per query. Id-overlap recall."""
    hits = 0
    for ids, want in zip(found, truth):
        hits += len(set(int(x) for x in ids[:k]) & set(int(x) for x in want[:k]))
    return hits / (len(truth) * k)


def percentile_ms(lat, p):
    return float(np.percentile(np.asarray(lat), p) * 1e3)


def single_thread_pass(search_one, queries):
    """One query per call, in order. Returns (results, per-query seconds, total seconds)."""
    results = [None] * len(queries)
    lat = np.empty(len(queries))
    clock = time.perf_counter
    start = clock()
    for i in range(len(queries)):
        t0 = clock()
        results[i] = search_one(queries[i])
        lat[i] = clock() - t0
    return results, lat, clock() - start


def threaded_pass(search_one, queries, threads):
    """Client-side thread pool issuing one query per call. Returns total seconds."""
    with ThreadPoolExecutor(threads) as pool:
        start = time.perf_counter()
        list(pool.map(search_one, queries, chunksize=64))
        return time.perf_counter() - start


def sweep(args, queries, truth, set_ef, search_one, all_cores_seconds, set_single=None, set_all=None):
    """Runs the ef sweep and returns the "search" list of the result schema.

    all_cores_seconds(queries) must run every query once using all cores and
    return the elapsed seconds.
    """
    rows = []
    for ef in [int(x) for x in args.efs.split(",")]:
        set_ef(ef)
        if set_single:
            set_single()
        best = None
        for _ in range(args.runs):
            results, lat, total = single_thread_pass(search_one, queries)
            if best is None or total < best[2]:
                best = (results, lat, total)
        results, lat, total = best
        if set_all:
            set_all()
        par = min(all_cores_seconds(queries) for _ in range(args.runs))
        row = {
            "ef": ef,
            "recall": round(recall_at_k(results, truth, args.k), 5),
            "qps_single": round(len(queries) / total, 1),
            "qps_all": round(len(queries) / par, 1),
            "threads_all": args.threads,
            "p50_ms": round(percentile_ms(lat, 50), 4),
            "p95_ms": round(percentile_ms(lat, 95), 4),
            "p99_ms": round(percentile_ms(lat, 99), 4),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def peak_rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def rss_bytes():
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
        return int(out.stdout.strip()) * 1024
    except (OSError, ValueError):
        return 0


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def system_info():
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": sys.platform,
        "arch": platform.machine(),
    }


def write_result(args, engine, meta, n, build, search, versions, notes, serialize=None, params=None):
    doc = {
        "engine": engine,
        "dataset": meta["dataset"],
        "metric": meta["metric"],
        "n": int(n),
        "dim": meta["dim"],
        "n_queries": meta["n_queries"],
        "k": args.k,
        "ground_truth": meta["ground_truth"],
        "params": params or {"m": args.m, "ef_construction": args.ef_construction},
        "build": build,
        "serialize": serialize,
        "search": search,
        "runs": args.runs,
        "system": system_info(),
        "versions": dict(versions, python=platform.python_version(), numpy=np.__version__),
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")
