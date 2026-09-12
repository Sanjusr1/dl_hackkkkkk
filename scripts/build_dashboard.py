"""Build a standalone HTML dashboard from benchmark outputs."""
from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
from pathlib import Path


LABELS = {
    "final_average_accuracy": "ACC",
    "average_incremental_accuracy": "AvgInc",
    "learning_accuracy": "LA",
    "backward_transfer": "BWT",
    "forgetting": "Forget",
    "final_macro_f1": "Macro-F1",
}


def pct(values: list[float]) -> str:
    vals = [float(v) for v in values if float(v) == float(v)]
    if not vals:
        return "n/a"
    mean = statistics.mean(vals) * 100
    std = statistics.stdev(vals) * 100 if len(vals) > 1 else 0.0
    return f"{mean:.1f} +/- {std:.1f}%"


def mean_metric(summary: dict, method: str, key: str) -> float:
    vals = [float(v) for v in summary[method].get(key, []) if float(v) == float(v)]
    return statistics.mean(vals) if vals else float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="runs/benchmark/summary.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--out", default="frontend/index.html")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    fig_dir = Path(args.figures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text())
    best_acc = max(summary, key=lambda m: mean_metric(summary, m, "final_average_accuracy"))
    lowest_forget = min(summary, key=lambda m: mean_metric(summary, m, "forgetting"))

    rows = []
    for method, metrics in summary.items():
        cells = "".join(f"<td>{pct(metrics.get(k, []))}</td>" for k in LABELS)
        rows.append(f"<tr><th>{html.escape(method)}</th>{cells}</tr>")

    figures = []
    for img in sorted(fig_dir.glob("*.png")):
        data = base64.b64encode(img.read_bytes()).decode("ascii")
        figures.append(
            f"""<figure>
  <img src="data:image/png;base64,{data}" alt="{html.escape(img.name)}">
  <figcaption>{html.escape(img.name)}</figcaption>
</figure>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DACL Results Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #172033; }}
    header {{ background: #172033; color: white; padding: 32px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    header p {{ margin: 0; color: #cbd5e1; }}
    main {{ padding: 28px 48px 48px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: white; border: 1px solid #dde3ee; border-radius: 8px; padding: 18px; }}
    .card span {{ display: block; color: #667085; font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde3ee; margin-bottom: 28px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #e8edf5; text-align: right; white-space: nowrap; }}
    th:first-child {{ text-align: left; }}
    thead {{ background: #edf2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #dde3ee; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 10px; color: #667085; font-size: 13px; }}
    @media (max-width: 760px) {{ header, main {{ padding-left: 18px; padding-right: 18px; }} .cards {{ grid-template-columns: 1fr; }} table {{ font-size: 12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Disaster-Adaptive Continual Learning</h1>
    <p>AIDERv2 class-incremental benchmark for measuring catastrophic forgetting.</p>
  </header>
  <main>
    <section class="cards">
      <div class="card"><span>Dataset</span><strong>AIDERv2</strong></div>
      <div class="card"><span>Best Final ACC</span><strong>{html.escape(best_acc)}</strong></div>
      <div class="card"><span>Lowest Forgetting</span><strong>{html.escape(lowest_forget)}</strong></div>
    </section>
    <h2>Benchmark Results</h2>
    <table>
      <thead><tr><th>Method</th>{''.join(f'<th>{v}</th>' for v in LABELS.values())}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <h2>Figures</h2>
    <section class="grid">{''.join(figures)}</section>
  </main>
</body>
</html>
"""
    out.write_text(page)
    print(f"dashboard -> {out}")


if __name__ == "__main__":
    main()
