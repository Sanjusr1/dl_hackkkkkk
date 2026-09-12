"""Evaluation over a task stream."""
from __future__ import annotations

import numpy as np
import torch

from ..data.scenario import make_loader
from .metrics import macro_f1, per_class_recall


@torch.no_grad()
def evaluate_task(method, dataset, cfg, device, num_classes: int) -> dict:
    """Accuracy / macro-F1 / confusion on one task's test set.

    Note this is *always* evaluated over the full label space seen so far --
    no task identity is handed to the model. Masking logits to the task's own
    classes would report "task-incremental" accuracy, which is far higher and
    not what a deployed system can do.
    """
    method.net.eval()
    loader = make_loader(dataset, cfg.train.eval_batch_size, shuffle=False, cfg=cfg)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = method.predict(x)
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        total += y.numel()
        for t, p in zip(y.tolist(), pred.tolist()):
            if t < num_classes and p < num_classes:
                conf[t, p] += 1

    method.net.train()
    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1(conf),
        "n": total,
        "confusion": conf,
        "recall": per_class_recall(conf),
    }


@torch.no_grad()
def evaluate_stream(method, scenario, cfg, device, upto: int | None = None,
                    include_future: bool = False) -> list[dict | None]:
    """Evaluate on every task up to `upto` (and optionally the unseen future)."""
    n_classes = method.net.num_classes
    results: list[dict | None] = []
    last = len(scenario) - 1 if include_future else (upto if upto is not None else len(scenario) - 1)
    for j, task in enumerate(scenario):
        if j > last:
            results.append(None)
            continue
        results.append(evaluate_task(method, task.test, cfg, device, n_classes))
    return results


def task_confusion(results: list[dict | None], scenario) -> np.ndarray:
    """How often task-j samples are predicted as *some* class of task k.

    The classic failure mode is not random error: everything collapses onto the
    most recent task's classes. This matrix makes that visible, and is the plot
    that best communicates forgetting to a non-ML audience.
    """
    T = len(scenario)
    out = np.zeros((T, T))
    owner = np.array(scenario_class_owner(scenario))
    for j, res in enumerate(results):
        if res is None:
            continue
        conf = res["confusion"]
        rows = [c for c in range(conf.shape[0]) if owner[c] == j] if len(owner) else []
        if not rows:
            continue
        block = conf[rows].sum(0)
        total = block.sum()
        if total == 0:
            continue
        for k in range(T):
            cols = [c for c in range(conf.shape[1]) if owner[c] == k]
            out[j, k] = block[cols].sum() / total
    return out


def scenario_class_owner(scenario) -> list[int]:
    """Global class id -> index of the task that introduced it."""
    owner = [0] * scenario.num_classes
    for t in scenario:
        for cid in t.class_ids:
            owner[cid] = t.index
    return owner
