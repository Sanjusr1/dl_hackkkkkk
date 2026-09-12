"""One-command project runner.

Examples:
    python run_project.py --fast
    python run_project.py --data /kaggle/input/datasets/banasmitajena/aiderv2-dataset --fast
    python run_project.py --full --seeds 0 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def find_aiderv2() -> Path | None:
    candidates = [
        Path("/kaggle/input/datasets/banasmitajena/aiderv2-dataset"),
        Path("/kaggle/input/aiderv2-dataset"),
    ]
    for c in candidates:
        if c.exists():
            return c
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("*aiderv2*"):
            if p.is_dir():
                return p
    return None


def prepare_kaggle_layout(src: Path) -> Path:
    """Normalize Kaggle's Train/Train, Val/Val, Test/Test layout."""
    mapping = {
        "train": [src / "Train" / "Train", src / "train" / "train", src / "Training" / "Training"],
        "val": [src / "Val" / "Val", src / "Validation" / "Validation", src / "valid" / "valid"],
        "test": [src / "Test" / "Test", src / "test" / "test", src / "Testing" / "Testing"],
    }
    if not any(choice.exists() for choices in mapping.values() for choice in choices):
        return src

    links = ROOT / "raw_aiderv2"
    if links.exists() or links.is_symlink():
        shutil.rmtree(links)
    links.mkdir(parents=True)

    for split, choices in mapping.items():
        target = next((p for p in choices if p.exists()), None)
        if target is None:
            raise FileNotFoundError(f"Could not find {split} split under {src}")
        os.symlink(target, links / split)
    return links


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DACL end to end with one command.")
    ap.add_argument("--data", default=None, help="Path to AIDERv2 input folder. Auto-detected on Kaggle.")
    ap.add_argument("--fast", action="store_true", help="Run the small AIDERv2 config for demo/time-limited runs.")
    ap.add_argument("--full", action="store_true", help="Run the full AIDERv2 config.")
    ap.add_argument("--synthetic", action="store_true", help="Run the built-in synthetic benchmark instead of AIDERv2.")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"])

    if args.synthetic:
        config = "configs/synthetic_cil.yaml"
        methods = args.methods or ["finetune", "er", "derpp", "icarl"]
    else:
        data = Path(args.data) if args.data else find_aiderv2()
        if data is None:
            raise SystemExit("AIDERv2 dataset not found. Pass --data /path/to/aiderv2-dataset.")
        normalized = prepare_kaggle_layout(data)
        run([
            sys.executable,
            "scripts/prepare_aiderv2.py",
            "--src",
            str(normalized),
            "--dst",
            "data/aiderv2",
            "--resize",
            "256",
            "--overwrite",
        ])
        config = "configs/aiderv2_cil.yaml" if args.full else "configs/aiderv2_cil_fast.yaml"
        if args.methods:
            methods = args.methods
        elif args.fast:
            methods = ["finetune", "derpp", "joint"]
        else:
            methods = ["finetune", "frozen", "ewc", "lwf", "er", "er_ace", "derpp", "icarl", "joint"]

    run([
        sys.executable,
        "scripts/run_benchmark.py",
        "--config",
        config,
        "--methods",
        *methods,
        "--seeds",
        *args.seeds,
    ])
    run([sys.executable, "scripts/plot_results.py", "--runs", "runs/benchmark/*", "--out", "figures/"])
    run([sys.executable, "scripts/build_dashboard.py", "--summary", "runs/benchmark/summary.json", "--figures", "figures", "--out", "frontend/index.html"])
    print("\nDone. Open frontend/index.html and use runs/benchmark/summary.json for the report.")


if __name__ == "__main__":
    main()
