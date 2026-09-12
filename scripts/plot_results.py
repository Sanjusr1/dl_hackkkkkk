"""Figures for the report.

    python scripts/plot_results.py --runs runs/benchmark/* --out figures/

Produces:
  accuracy_matrix_<run>.png  heatmap of R -- the single figure that shows what
                             a method does to the past
  retention_<run>.png        per-task accuracy against stream position
  comparison.png             ACC / BWT / forgetting across methods
  task_confusion_<run>.png   where old-task samples get sent instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(run_dir: Path) -> dict | None:
    f = run_dir / "results.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def plot_matrix(res: dict, out: Path, key: str = "accuracy_matrix", title: str | None = None):
    R = np.array(res[key], dtype=float)
    T = R.shape[0]
    fig, ax = plt.subplots(figsize=(1.1 * T + 2.4, 1.1 * T + 1.8))
    masked = np.ma.masked_invalid(R)
    im = ax.imshow(masked * 100, cmap="viridis", vmin=0, vmax=100)
    for i in range(T):
        for j in range(T):
            if np.isnan(R[i, j]):
                continue
            ax.text(j, i, f"{100 * R[i, j]:.0f}", ha="center", va="center",
                    color="white" if R[i, j] < 0.6 else "black", fontsize=9)
    ax.set_xticks(range(T), [f"T{j}" for j in range(T)])
    ax.set_yticks(range(T), [f"after T{i}" for i in range(T)])
    ax.set_xlabel("evaluated on task")
    ax.set_title(title or f"{res.get('run_name', '')} - accuracy (%)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_retention(res: dict, out: Path):
    R = np.array(res["accuracy_matrix"], dtype=float)
    T = R.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for j in range(T):
        steps = list(range(j, T))
        ax.plot(steps, [100 * R[i, j] for i in steps], marker="o", label=f"task {j}")
    ax.set_xlabel("tasks trained so far")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 101)
    ax.set_xticks(range(T))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title(f"{res.get('run_name', '')} - retention of each task")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_task_confusion(res: dict, out: Path):
    if "task_confusion" not in res:
        return
    C = np.array(res["task_confusion"], dtype=float)
    T = C.shape[0]
    fig, ax = plt.subplots(figsize=(1.0 * T + 2.4, 1.0 * T + 1.6))
    im = ax.imshow(C * 100, cmap="magma", vmin=0, vmax=100)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{100 * C[i, j]:.0f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.6 else "black", fontsize=9)
    ax.set_xlabel("predicted as a class of task")
    ax.set_ylabel("true task")
    ax.set_xticks(range(T), [f"T{j}" for j in range(T)])
    ax.set_yticks(range(T), [f"T{i}" for i in range(T)])
    ax.set_title("where old-task samples go (final model, %)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_comparison(all_res: list[dict], out: Path):
    by_method: dict[str, list[dict]] = {}
    for r in all_res:
        by_method.setdefault(r["config"]["method"]["name"], []).append(r)
    if not by_method:
        return

    metrics = [("final_average_accuracy", "final avg accuracy"),
               ("backward_transfer", "backward transfer"),
               ("forgetting", "forgetting")]
    methods = list(by_method)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    for ax, (key, label) in zip(np.atleast_1d(axes), metrics):
        means, errs = [], []
        for m in methods:
            vals = [100 * r["metrics"][key] for r in by_method[m]]
            means.append(np.mean(vals))
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax.bar(range(len(methods)), means, yerr=errs, capsize=3,
               color=["#c44"] * len(methods) if key != "final_average_accuracy" else None)
        ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{label} (%)")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Continual-learning strategies on the disaster stream")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = []
    for d in args.runs:
        p = Path(d)
        res = load(p)
        if res is None:
            print(f"[skip] no results.json in {p}")
            continue
        loaded.append(res)
        name = res.get("run_name", p.name)
        plot_matrix(res, out / f"accuracy_matrix_{name}.png")
        plot_retention(res, out / f"retention_{name}.png")
        plot_task_confusion(res, out / f"task_confusion_{name}.png")
    if loaded:
        plot_comparison(loaded, out / "comparison.png")
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
