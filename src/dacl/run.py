"""Run one experiment.

    python -m dacl.run --config configs/synthetic_cil.yaml --method derpp --seed 0
    python -m dacl.run --config configs/synthetic_cil.yaml \
        --set method.params.buffer_size=1000 train.epochs=3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import Config
from .data.scenario import build_scenario
from .engine.trainer import ContinualTrainer
from .utils import get_device, save_json, set_seed, setup_logging


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Disaster-Adaptive Continual Learning")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--method", type=str, default=None, help="shortcut for method.name")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[],
                    help="dotted overrides, e.g. train.epochs=3 method.params.alpha=0.2")
    return ap.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config.load(args.config)
    if args.method:
        cfg.method.name = args.method
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tag:
        cfg.tag = args.tag
    cfg.apply_overrides(args.overrides)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    return cfg


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = build_config(args)

    out_dir = Path(cfg.output_dir) / cfg.run_name
    logger = setup_logging(out_dir)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    logger.info(f"run={cfg.run_name} device={device}")
    scenario = build_scenario(cfg)
    logger.info(scenario.summary())

    t0 = time.time()
    trainer = ContinualTrainer(cfg, scenario, device, logger)
    results = trainer.run()
    results["wall_time_sec"] = time.time() - t0
    results["run_name"] = cfg.run_name

    save_json(results, out_dir / "results.json")
    np.savetxt(out_dir / "accuracy_matrix.csv", trainer.R, delimiter=",", fmt="%.6f")
    logger.info(f"wrote {out_dir / 'results.json'} ({results['wall_time_sec']:.1f}s)")
    return results


if __name__ == "__main__":
    main()
