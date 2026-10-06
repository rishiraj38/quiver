"""Download an ann-benchmarks HDF5 dataset and convert it to fvecs/ivecs.

Output layout (under bench/datasets/<dataset>/):
  base.fvecs         train vectors
  query.fvecs        test vectors
  groundtruth.ivecs  true neighbour ids, 100 per query
  meta.json          metric, counts, dimension
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://ann-benchmarks.com/{name}.hdf5"


def download(url, dest, attempts=5):
    """Fetch url to dest, resuming a partial file across retries."""
    part = dest + ".part"
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        # The host rejects the default Python-urllib user agent with a 403.
        req = urllib.request.Request(url, headers={"User-Agent": "quiver-bench/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                last = time.time()
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 5:
                            last = time.time()
                            print(f"  {have / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
            if total and have < total:
                raise IOError(f"short read: {have} of {total} bytes")
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                os.remove(part)
            print(f"download attempt {attempt} failed: {e}", file=sys.stderr)
        except (OSError, urllib.error.URLError) as e:
            print(f"download attempt {attempt} failed: {e}", file=sys.stderr)
        time.sleep(3 * attempt)
    raise SystemExit(f"could not download {url}")


def write_vecs(path, arr, dtype):
    """Write rows as <int32 dim><dim values>, the fvecs/ivecs layout."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    n, d = arr.shape
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        step = 100_000
        for i in range(0, n, step):
            block = arr[i : i + step]
            out = np.empty((len(block), d + 1), dtype="<i4")
            out[:, 0] = d
            out[:, 1:] = block.view("<i4")
            out.tofile(f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--dir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--force", action="store_true", help="reconvert even if outputs exist")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    h5 = os.path.join(args.dir, args.dataset + ".hdf5")
    out = os.path.join(args.dir, args.dataset)

    if os.path.exists(h5):
        print(f"{h5} present, skipping download")
    else:
        url = URL.format(name=args.dataset)
        print(f"downloading {url}")
        download(url, h5)

    names = ["base.fvecs", "query.fvecs", "groundtruth.ivecs", "meta.json"]
    if not args.force and all(os.path.exists(os.path.join(out, n)) for n in names):
        print(f"{out} already converted")
        return

    os.makedirs(out, exist_ok=True)
    with h5py.File(h5, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
        neighbors = f["neighbors"][:]
        distance = f.attrs.get("distance", "euclidean")
        if isinstance(distance, bytes):
            distance = distance.decode()

    write_vecs(os.path.join(out, "base.fvecs"), train, "<f4")
    write_vecs(os.path.join(out, "query.fvecs"), test, "<f4")
    write_vecs(os.path.join(out, "groundtruth.ivecs"), neighbors, "<i4")

    metric = {"euclidean": "l2", "angular": "cosine"}.get(str(distance), str(distance))
    meta = {
        "dataset": args.dataset,
        "metric": metric,
        "n": int(train.shape[0]),
        "dim": int(train.shape[1]),
        "n_queries": int(test.shape[0]),
        "gt_k": int(neighbors.shape[1]),
        "source": URL.format(name=args.dataset),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
