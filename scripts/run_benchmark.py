"""Sweep methods x seeds and print the comparison table that goes in the report.

    python scripts/run_benchmark.py --config configs/synthetic_cil.yaml \
        --methods finetune ewc lwf er er_ace derpp icarl joint --seeds 0 1 2

Single-seed continual-learning results are close to meaningless -- class order
alone swings final accuracy by several points -- so everything is reported as
mean +/- std over seeds.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dacl.run import main as run_one  # noqa: E402

HEADLINE = [
    ("final_average_accuracy", "ACC", True),
    ("average_incremental_accuracy", "AvgInc", True),
    ("learning_accuracy", "LA", True),
    ("backward_transfer", "BWT", True),
    ("forgetting", "Forget", True),
    ("final_macro_f1", "MacroF1", True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="+", default=["finetune", "ewc", "lwf", "er", "derpp", "joint"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output-dir", default="runs/benchmark")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    table: dict[str, dict[str, list[float]]] = {}
    for method in args.methods:
        table[method] = {k: [] for k, _, _ in HEADLINE}
        for seed in args.seeds:
            argv = ["--config", args.config, "--method", method, "--seed", str(seed),
                    "--output-dir", args.output_dir]
            if args.overrides:
                argv += ["--set", *args.overrides]
            res = run_one(argv)
            for key, _, _ in HEADLINE:
                table[method][key].append(float(res["metrics"].get(key, float("nan"))))

    out = Path(args.output_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))

    header = "method".ljust(12) + "".join(label.rjust(16) for _, label, _ in HEADLINE)
    print("\n" + header)
    print("-" * len(header))
    for method, vals in table.items():
        row = method.ljust(12)
        for key, _, as_pct in HEADLINE:
            xs = [v for v in vals[key] if v == v]
            if not xs:
                row += "n/a".rjust(16)
                continue
            mu = statistics.mean(xs) * (100 if as_pct else 1)
            sd = (statistics.stdev(xs) if len(xs) > 1 else 0.0) * (100 if as_pct else 1)
            row += f"{mu:.1f}+/-{sd:.1f}".rjust(16)
        print(row)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
