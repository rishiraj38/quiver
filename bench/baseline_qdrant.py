"""Qdrant baseline. Starts its own container, measures over gRPC, removes it.

Only the container named by --container and the directory given by --storage
are ever created or removed.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib.metadata import version

import numpy as np
from qdrant_client import QdrantClient, models

import common

COLLECTION = "bench"


def docker(*argv, check=True):
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=check)


def docker_available():
    try:
        return docker("info", "--format", "{{.NCPU}} {{.MemTotal}}", check=False)
    except FileNotFoundError:
        return None


def start_container(args):
    if docker("ps", "-aq", "--filter", f"name=^{args.container}$").stdout.strip():
        raise SystemExit(f"container {args.container} already exists; remove it first")
    os.makedirs(args.storage, exist_ok=True)
    docker(
        "run", "-d",
        "--name", args.container,
        "--memory", args.memory,
        "-p", f"127.0.0.1:{args.http_port}:6333",
        "-p", f"127.0.0.1:{args.grpc_port}:6334",
        "-v", f"{os.path.abspath(args.storage)}:/qdrant/storage",
        args.image,
    )  # fmt: skip
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.http_port}/readyz", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise SystemExit("qdrant did not become ready within 120s")


def stop_container(args):
    docker("rm", "-f", args.container, check=False)
    shutil.rmtree(args.storage, ignore_errors=True)


def wait_indexed(client, n, timeout=3600):
    """Block until the optimiser is idle and every vector is in an HNSW segment."""
    deadline = time.time() + timeout
    stable = 0
    while time.time() < deadline:
        info = client.get_collection(COLLECTION)
        done = info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n
        stable = stable + 1 if done else 0
        if stable >= 3:
            return info
        time.sleep(1)
    raise SystemExit(f"indexing did not finish within {timeout}s (status {info.status})")


def run(args, client, base, queries, truth, meta, docker_info):
    n, dim = base.shape
    distance = models.Distance.EUCLID if meta["metric"] == "l2" else models.Distance.COSINE
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=distance),
        hnsw_config=models.HnswConfigDiff(m=args.m, ef_construct=args.ef_construction),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=args.segments),
    )

    t0 = time.perf_counter()
    client.upload_collection(COLLECTION, vectors=base, ids=range(n), batch_size=1024, wait=True)
    upload_s = time.perf_counter() - t0
    print(f"upload {upload_s:.1f}s, waiting for indexing", flush=True)
    info = wait_indexed(client, n)
    build_s = time.perf_counter() - t0
    print(f"build {build_s:.1f}s, {info.segments_count} segments", flush=True)

    build = {
        "seconds": round(build_s, 2),
        "upload_seconds": round(upload_s, 2),
        "threads": docker_info.get("ncpu"),
        "segments": info.segments_count,
        "container_mem_usage": docker(
            "stats", "--no-stream", "--format", "{{.MemUsage}}", args.container, check=False
        ).stdout.strip(),
    }

    state = {"params": None}

    def set_ef(ef):
        state["params"] = models.SearchParams(hnsw_ef=ef)

    def search_one(q):
        res = client.query_points(
            COLLECTION, query=q, limit=args.k, search_params=state["params"], with_payload=False
        )
        return [p.id for p in res.points]

    def all_cores(qs):
        return common.threaded_pass(search_one, qs, args.threads)

    search = common.sweep(args, queries, truth, set_ef, search_one, all_cores)

    try:
        server = client.info().version
    except Exception:
        server = "unknown"
    notes = [
        "Qdrant runs in a Docker container and is queried through its gRPC client, so every "
        "query pays serialisation, a loopback network round trip and the Python client. The "
        "in-process engines pay none of that; compare recall freely, QPS with that in mind.",
        f"Container limits: --memory {args.memory}; Docker VM has {docker_info.get('ncpu')} CPUs.",
        "build seconds = upload start until the collection is green with all vectors indexed.",
        "qps_single: one client thread, one request per query, sequential.",
        "qps_all: one request per query from a Python thread pool; the client holds the GIL "
        "between requests, so this understates what the server can do.",
        "recall is id overlap with the true top-k; ties in distance are not credited.",
    ]
    versions = {"qdrant": server, "qdrant_client": version("qdrant-client"), "image": args.image}
    params = {"m": args.m, "ef_construction": args.ef_construction}
    common.write_result(args, "qdrant", meta, n, build, search, versions, notes, None, params)


def main():
    ap = common.base_parser("qdrant")
    ap.add_argument("--image", default="qdrant/qdrant:latest")
    ap.add_argument("--container", default="quiver-bench-qdrant")
    ap.add_argument("--memory", default="5g")
    ap.add_argument("--storage", default=os.path.join(common.HERE, "qdrant_storage"))
    ap.add_argument("--http-port", type=int, default=6333)
    ap.add_argument("--grpc-port", type=int, default=6334)
    ap.add_argument("--segments", type=int, default=1, help="target segment count; 0 = Qdrant default")
    ap.add_argument("--keep", action="store_true", help="leave the container running afterwards")
    args = ap.parse_args()

    probe = docker_available()
    if probe is None or probe.returncode != 0:
        raise SystemExit("docker is not running; start it and rerun (nothing was measured)")
    ncpu, mem = (probe.stdout.split() + ["", ""])[:2]
    docker_info = {"ncpu": int(ncpu) if ncpu.isdigit() else None, "mem_total": mem}

    base, queries, truth, meta = common.load_dataset(args)
    start_container(args)
    try:
        client = QdrantClient(
            host="127.0.0.1", port=args.http_port, grpc_port=args.grpc_port, prefer_grpc=True, timeout=120
        )
        run(args, client, base, queries, truth, meta, docker_info)
    finally:
        if not args.keep:
            stop_container(args)


if __name__ == "__main__":
    main()
