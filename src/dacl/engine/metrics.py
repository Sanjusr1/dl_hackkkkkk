"""Continual-learning metrics.

Everything derives from the accuracy matrix R, where R[i, j] is accuracy on
task j's test set after finishing training on task i. Rows = time, columns =
task. The upper triangle (j > i) is only meaningful for forward transfer.

Definitions follow Lopez-Paz & Ranzato (GEM, 2017) and Chaudhry et al. (2018)
so the numbers are comparable with the published literature.
"""
from __future__ import annotations

import numpy as np


def final_average_accuracy(R: np.ndarray) -> float:
    """ACC: mean accuracy over all tasks once the whole stream has been seen."""
    last = R[-1, : R.shape[1]]
    return float(np.nanmean(last))


def average_incremental_accuracy(R: np.ndarray) -> float:
    """Mean over time of the accuracy on tasks seen so far.

    Rewards a method for being useful *throughout* the stream, not only at the
    end -- which is what matters in a response operation, where the model is
    queried after every event, not once at the end of the season.
    """
    T = R.shape[0]
    per_step = [np.nanmean(R[i, : i + 1]) for i in range(T)]
    return float(np.mean(per_step))


def backward_transfer(R: np.ndarray) -> float:
    """BWT: mean change on old tasks between learning them and the end.

    Negative = forgetting; positive = learning later tasks helped earlier ones.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    diffs = [R[-1, i] - R[i, i] for i in range(T - 1)]
    return float(np.mean(diffs))


def forgetting(R: np.ndarray) -> float:
    """Average maximum drop from a task's best-ever accuracy to its final one.

    Stricter than BWT: it charges a method for transient collapse even if a
    later task accidentally recovers the score.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    out = []
    for j in range(T - 1):
        best = np.nanmax(R[j:T - 1, j])
        out.append(best - R[-1, j])
    return float(np.mean(out))


def forward_transfer(R: np.ndarray, random_baseline: np.ndarray | None = None) -> float:
    """FWT: accuracy on a future task before training on it, vs. the untrained model."""
    T = R.shape[0]
    if T < 2 or random_baseline is None:
        return float("nan")
    vals = [R[i - 1, i] - random_baseline[i] for i in range(1, T)
            if not np.isnan(R[i - 1, i])]
    return float(np.mean(vals)) if vals else float("nan")


def learning_accuracy(R: np.ndarray) -> float:
    """Mean of the diagonal: how well each task is learned when it arrives.

    Separates the two ways a method can fail -- it may forget (low BWT) or it
    may be too rigid to learn the new disaster at all (low LA). EWC with a large
    lambda fails the second way, and average accuracy alone hides that.
    """
    return float(np.nanmean(np.diag(R)))


def stability_plasticity(R: np.ndarray) -> dict[str, float]:
    return {
        "plasticity_learning_accuracy": learning_accuracy(R),
        "stability_backward_transfer": backward_transfer(R),
    }


def summarize(R: np.ndarray, random_baseline: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "final_average_accuracy": final_average_accuracy(R),
        "average_incremental_accuracy": average_incremental_accuracy(R),
        "backward_transfer": backward_transfer(R),
        "forgetting": forgetting(R),
        "learning_accuracy": learning_accuracy(R),
        "forward_transfer": forward_transfer(R, random_baseline),
        "last_task_accuracy": float(R[-1, -1]),
    }
    out.update(stability_plasticity(R))
    return out


# ------------------------------------------------------- per-class diagnostics
def macro_f1(confusion: np.ndarray) -> float:
    """Macro-F1 over classes present in the confusion matrix.

    Reported alongside accuracy because disaster class frequencies are wildly
    imbalanced (undamaged tiles dominate every real corpus), and a model that
    quietly stops predicting a rare class keeps a respectable accuracy.
    """
    f1s = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        if tp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def per_class_recall(confusion: np.ndarray) -> np.ndarray:
    support = confusion.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(support > 0, np.diag(confusion) / np.maximum(support, 1), np.nan)
    return rec
