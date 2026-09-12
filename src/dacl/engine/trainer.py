"""The continual training loop.

One pass over the disaster stream:

    for each task t:
        grow the head by the new classes
        (re)build the optimiser
        train for E epochs, loss supplied by the method
        method.end_task -> consolidate (Fisher / teacher / exemplars)
        evaluate on tasks 0..t  (and t+1.. for forward transfer)

Every method sees the identical stream, augmentation and optimiser, so the only
thing that differs between rows of the results table is the strategy itself.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..data.scenario import concat_train, make_loader
from ..methods import build_method
from ..models.network import DACLNet
from ..utils import AverageMeter, format_matrix
from .evaluator import evaluate_stream, task_confusion
from .metrics import summarize


def build_optimizer(net, cfg):
    groups = net.param_groups(cfg.train.lr, cfg.train.backbone_lr_scale)
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.train.lr, momentum=cfg.train.momentum,
                               weight_decay=cfg.train.weight_decay, nesterov=True)
    return torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.train.epochs * steps_per_epoch))
    if cfg.train.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.train.epochs // 3), gamma=0.1)
    return None


class ContinualTrainer:
    def __init__(self, cfg, scenario, device, logger):
        self.cfg = cfg
        self.scenario = scenario
        self.device = device
        self.log = logger

        self.net = DACLNet(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            head=cfg.model.head,
            dropout=cfg.model.dropout,
            freeze_backbone=cfg.model.freeze_backbone,
        ).to(device)
        self.method = build_method(cfg, self.net, device)

        T = len(scenario)
        self.R = np.full((T, T), np.nan)           # accuracy matrix
        self.F1 = np.full((T, T), np.nan)          # macro-F1, same layout
        self.random_baseline = np.full(T, np.nan)  # untrained accuracy per task
        self.history: list[dict] = []

    # ------------------------------------------------------------------- setup
    def _grow_head(self, task) -> None:
        if self.scenario.kind == "domain_incremental":
            if self.net.num_classes == 0:
                self.net.expand(self.scenario.num_classes, 0)
        else:
            self.net.expand(task.num_classes, task.index)
        self.net.to(self.device)

    def _measure_random_baseline(self) -> None:
        """Accuracy of the untrained (but head-complete) model, for forward transfer."""
        probe = DACLNet(
            backbone=self.cfg.model.backbone, pretrained=self.cfg.model.pretrained,
            head=self.cfg.model.head, dropout=self.cfg.model.dropout,
        )
        probe.expand(self.scenario.num_classes, 0)
        probe.to(self.device)
        from ..methods.baselines import Finetune
        probe_method = Finetune(probe, self.cfg, self.device)
        for j, task in enumerate(self.scenario):
            from .evaluator import evaluate_task
            res = evaluate_task(probe_method, task.test, self.cfg, self.device,
                                self.scenario.num_classes)
            self.random_baseline[j] = res["accuracy"]
        del probe, probe_method

    # ------------------------------------------------------------------- train
    def train_task(self, task) -> dict:
        cfg = self.cfg
        if self.method.wants_cumulative_data:
            dataset = concat_train(self.scenario.tasks[: task.index + 1])
        else:
            dataset = task.train
        loader = make_loader(dataset, cfg.train.batch_size, shuffle=True, cfg=cfg)

        self.method.begin_task(task, self.scenario, loader)
        optimizer = build_optimizer(self.net, cfg)
        scheduler = build_scheduler(optimizer, cfg, max(1, len(loader)))

        self.net.train()
        stats = {}
        for epoch in range(cfg.train.epochs):
            meters: dict[str, AverageMeter] = {}
            correct = total = 0
            t0 = time.time()
            for x, y, x_raw in loader:
                x, y, x_raw = x.to(self.device), y.to(self.device), x_raw.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = self.method.observe(x, y, x_raw)
                loss.backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.train.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    pred = self.net(x).argmax(1)
                    correct += int((pred == y).sum())
                    total += y.numel()
                parts["loss"] = float(loss.detach())
                for k, v in parts.items():
                    meters.setdefault(k, AverageMeter()).update(v, y.size(0))

            stats = {k: m.avg for k, m in meters.items()}
            stats["train_acc"] = correct / max(total, 1)
            self.log.info(
                f"  task {task.index} epoch {epoch + 1}/{cfg.train.epochs} "
                + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                + f" ({time.time() - t0:.1f}s)"
            )

        # Head-bias correction for class-incremental linear heads.
        if (self.scenario.kind == "class_incremental" and task.index > 0
                and self.cfg.model.head == "linear"):
            self.net.head.weight_align(task.num_classes)

        self.method.end_task(task, self.scenario, loader)
        return stats

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        if self.cfg.eval_future_tasks:
            self._measure_random_baseline()

        for task in self.scenario:
            self.log.info(f"=== {task} ===")
            self._grow_head(task)
            train_stats = self.train_task(task)

            results = evaluate_stream(
                self.method, self.scenario, self.cfg, self.device,
                upto=task.index, include_future=self.cfg.eval_future_tasks,
            )
            for j, res in enumerate(results):
                if res is None:
                    continue
                if j > task.index and not self.cfg.eval_future_tasks:
                    continue
                self.R[task.index, j] = res["accuracy"]
                self.F1[task.index, j] = res["macro_f1"]

            seen = self.R[task.index, : task.index + 1]
            self.log.info(
                f"  after task {task.index}: seen-avg acc={100 * np.nanmean(seen):.2f}%  "
                + " ".join(f"T{j}={100 * self.R[task.index, j]:.1f}" for j in range(task.index + 1))
            )
            self.history.append({
                "task": task.index,
                "name": task.name,
                "train": train_stats,
                "eval": [
                    None if r is None else
                    {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n": r["n"]}
                    for r in results
                ],
                "method_state": self.method.extra_state(),
            })
            self.last_results = results

        labels = [f"T{t.index}" for t in self.scenario]
        self.log.info("Accuracy matrix R[i,j] = acc on task j after training task i (%)\n"
                      + format_matrix(self.R, labels, labels))

        metrics = summarize(self.R, self.random_baseline)
        metrics["final_macro_f1"] = float(np.nanmean(self.F1[-1]))
        self.log.info("  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        out = {
            "config": self.cfg.to_dict(),
            "scenario": {
                "kind": self.scenario.kind,
                "classes": self.scenario.classes,
                "tasks": [{"index": t.index, "name": t.name, "classes": t.class_names,
                           "train_size": len(t.train), "test_size": len(t.test)}
                          for t in self.scenario],
            },
            "accuracy_matrix": self.R.tolist(),
            "macro_f1_matrix": self.F1.tolist(),
            "random_baseline": self.random_baseline.tolist(),
            "metrics": metrics,
            "history": self.history,
        }
        if self.scenario.kind == "class_incremental":
            out["task_confusion"] = task_confusion(self.last_results, self.scenario).tolist()
            out["final_confusion"] = [
                None if r is None else r["confusion"].tolist() for r in self.last_results
            ]
        return out
