"""Draw recall@k vs single-thread QPS from bench/results/*.json and print a markdown table."""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

# Colour belongs to the engine, not to its position in the legend, so a
# missing result file never repaints the others.
ENGINES = [
    ("quiver", "quiver", "#2a78d6"),
    ("faiss", "FAISS", "#eb6834"),
    ("hnswlib", "hnswlib", "#1baf7a"),
    ("qdrant", "Qdrant (gRPC)", "#eda100"),
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def load(results_dir):
    docs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            doc = json.load(f)
        if "engine" in doc and doc.get("search"):
            docs[doc["engine"]] = doc
    return docs


def plot(docs, out_base):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    first = next(iter(docs.values()))
    k = first["k"]
    for key, label, colour in ENGINES:
        doc = docs.get(key)
        if not doc:
            continue
        rows = sorted(doc["search"], key=lambda r: r["recall"])
        xs = [r["recall"] for r in rows]
        ys = [r["qps_single"] for r in rows]
        ax.plot(
            xs, ys,
            color=colour, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3,
        )  # fmt: skip
        # Name each line at its low-recall end, where the curves are furthest apart.
        ax.annotate(
            label, (xs[0], ys[0]), xytext=(-8, 0), textcoords="offset points",
            ha="right", va="center", fontsize=9, color=INK_SECONDARY,
        )  # fmt: skip

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, which="both", labelsize=9, length=3)
    lo = min(r["recall"] for d in docs.values() for r in d["search"])
    ax.set_xlim(lo - 0.045, 1.004)

    ax.set_xlabel(f"recall@{k}", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("queries per second, one thread (log scale)", color=INK_SECONDARY, fontsize=10)
    params = first.get("params", {})
    fig.suptitle(
        f"Recall vs throughput on {first['dataset']}",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )  # fmt: skip
    ax.set_title(
        f"{first['n']:,} vectors, HNSW M={params.get('m')}, efConstruction={params.get('ef_construction')}, "
        f"one query per call. {first['system']['cpu']}. Up and to the right is better.",
        loc="left", fontsize=8.5, color=INK_SECONDARY, pad=10,
    )  # fmt: skip
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 1.03))
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", facecolor=SURFACE)
        print(f"wrote {out_base}.{ext}")
    plt.close(fig)


def table(docs, efs):
    head = ["engine", "version", "build (s)"]
    for ef in efs:
        head += [f"recall@{next(iter(docs.values()))['k']} ef={ef}", f"QPS ef={ef}"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for key, label, _ in ENGINES:
        doc = docs.get(key)
        if not doc:
            continue
        ver = doc.get("versions", {}).get(key) or doc.get("versions", {}).get("go", "")
        cells = [label, str(ver), f"{doc['build']['seconds']:.1f}"]
        by_ef = {r["ef"]: r for r in doc["search"]}
        for ef in efs:
            r = by_ef.get(ef)
            cells += [f"{r['recall']:.4f}", f"{r['qps_single']:,.0f}"] if r else ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--table-efs", default="40,120", help="ef values shown in the markdown table")
    args = ap.parse_args()

    docs = load(args.results)
    if not docs:
        raise SystemExit(f"no result files in {args.results}")
    shapes = {(d["dataset"], d["n"], d["k"]) for d in docs.values()}
    if len(shapes) > 1:
        raise SystemExit(f"results are not comparable, they cover different runs: {sorted(shapes)}")

    plot(docs, os.path.join(args.results, "recall_qps"))
    print()
    print(table(docs, [int(x) for x in args.table_efs.split(",")]))
    print()
    print("QPS is single-thread, one query per call.")


if __name__ == "__main__":
    main()
