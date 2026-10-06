# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=3", "datasets>=2.19", "requests"]
# ///
"""Semantic search over news headlines, served by quiver.

    ./bin/quiver -data ./data &
    uv run examples/semantic_search.py --index          # embed and load 20k articles
    uv run examples/semantic_search.py "rate cut by the central bank"
    uv run examples/semantic_search.py "rate cut by the central bank" --topic Sci/Tech
"""
import argparse

import requests

URL = "http://localhost:6333"
NAME = "news"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOPICS = ["World", "Sports", "Business", "Sci/Tech"]


def index(n: int) -> None:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rows = load_dataset("fancyzhx/ag_news", split=f"train[:{n}]")
    model = SentenceTransformer(MODEL)
    vectors = model.encode(rows["text"], batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    requests.delete(f"{URL}/collections/{NAME}")
    r = requests.put(f"{URL}/collections/{NAME}", json={"dim": vectors.shape[1], "metric": "cosine"})
    r.raise_for_status()
    for start in range(0, n, 1000):
        points = [
            {
                "id": i,
                "vector": vectors[i].tolist(),
                "payload": {"text": rows[i]["text"], "topic": TOPICS[rows[i]["label"]]},
            }
            for i in range(start, min(start + 1000, n))
        ]
        requests.post(f"{URL}/collections/{NAME}/points", json={"points": points}).raise_for_status()
    print(requests.get(f"{URL}/collections/{NAME}").json())


def search(query: str, topic: str | None, k: int) -> None:
    from sentence_transformers import SentenceTransformer

    vector = SentenceTransformer(MODEL).encode(query, normalize_embeddings=True).tolist()
    body = {"vector": vector, "k": k}
    if topic:
        body["filter"] = {"must": [{"key": "topic", "op": "eq", "value": topic}]}
    r = requests.post(f"{URL}/collections/{NAME}/search", json=body)
    r.raise_for_status()
    out = r.json()
    print(f"{len(out['hits'])} hits in {out['took_us']} us")
    for h in out["hits"]:
        print(f"  {h['distance']:.3f}  [{h['payload']['topic']}]  {h['payload']['text'][:110]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("--topic", choices=TOPICS)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    if args.index:
        index(args.n)
    if args.query:
        search(args.query, args.topic, args.k)
