"""Episodic memory.

Two population strategies, because they answer different questions:

Reservoir (ER, DER++)
    Online, single pass, no knowledge of task boundaries. Each of the N samples
    seen so far is in memory with equal probability -- the right default when
    imagery streams in continuously during a response operation.

Herding (iCaRL)
    Offline at the task boundary. Greedily picks the exemplars whose running
    feature mean best tracks the true class mean, then shrinks the per-class
    quota as new classes arrive so total memory stays constant. Memory is a hard
    constraint in the field (an edge box on a response laptop), so a fixed
    *total* budget is the honest comparison, not a fixed per-class one.
"""
from __future__ import annotations

import numpy as np
import torch


class ReservoirBuffer:
    """Fixed-capacity memory of raw (un-augmented) tensors."""

    def __init__(self, capacity: int, device: torch.device, store_logits: bool = False):
        self.capacity = int(capacity)
        self.device = device
        self.store_logits = store_logits
        self.n_seen = 0
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None
        self.t: torch.Tensor | None = None

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        if self.y is None:
            return 0
        return int(min(self.n_seen, self.capacity))

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def _init_storage(self, x, n_logits: int) -> None:
        c, h, w = x.shape[1:]
        self.x = torch.zeros(self.capacity, c, h, w, device=self.device)
        self.y = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.t = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        if self.store_logits:
            self.logits = torch.zeros(self.capacity, n_logits, device=self.device)

    # ------------------------------------------------------------------- write
    @torch.no_grad()
    def add(self, x, y, logits=None, task_id: int = -1) -> None:
        if self.capacity == 0:
            return
        if self.x is None:
            self._init_storage(x, logits.shape[1] if logits is not None else 0)

        for i in range(x.shape[0]):
            slot = self._reservoir_slot()
            self.n_seen += 1
            if slot < 0:
                continue
            self.x[slot] = x[i].detach().to(self.device)
            self.y[slot] = int(y[i])
            self.t[slot] = task_id
            if self.store_logits and logits is not None:
                row = logits[i].detach().to(self.device)
                self.logits[slot].zero_()
                self.logits[slot, : row.numel()] = row

    def _reservoir_slot(self) -> int:
        if self.n_seen < self.capacity:
            return self.n_seen
        j = np.random.randint(0, self.n_seen + 1)
        return j if j < self.capacity else -1

    @torch.no_grad()
    def grow_logits(self, n_classes: int) -> None:
        """The label space expanded; pad stored logits with -inf-ish zeros."""
        if not self.store_logits or self.logits is None:
            return
        if n_classes <= self.logits.shape[1]:
            return
        pad = torch.zeros(self.capacity, n_classes - self.logits.shape[1], device=self.device)
        self.logits = torch.cat([self.logits, pad], dim=1)

    # -------------------------------------------------------------------- read
    @torch.no_grad()
    def sample(self, batch_size: int):
        n = len(self)
        if n == 0:
            return None
        idx = torch.from_numpy(
            np.random.choice(n, size=min(batch_size, n), replace=False)
        ).to(self.device)
        out = {"x": self.x[idx], "y": self.y[idx], "t": self.t[idx]}
        if self.store_logits and self.logits is not None:
            out["logits"] = self.logits[idx]
        return out

    def state(self) -> dict:
        return {"capacity": self.capacity, "filled": len(self), "seen": self.n_seen}


class HerdingMemory:
    """Class-balanced exemplar set with a fixed global budget (iCaRL)."""

    def __init__(self, budget: int, device: torch.device):
        self.budget = int(budget)
        self.device = device
        self.by_class: dict[int, torch.Tensor] = {}     # class id -> (m, C, H, W)
        self.means: dict[int, torch.Tensor] = {}        # class id -> normalised mean feature

    def __len__(self) -> int:
        return sum(v.shape[0] for v in self.by_class.values())

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def quota(self, n_classes: int) -> int:
        return max(1, self.budget // max(1, n_classes))

    @torch.no_grad()
    def update(self, net, dataset, new_class_ids: list[int], batch_size: int = 64) -> None:
        """Select exemplars for the new classes, then trim the old ones."""
        n_total = len(self.by_class) + len(new_class_ids)
        m = self.quota(n_total)

        for cid, tensor in list(self.by_class.items()):        # shrink old classes
            self.by_class[cid] = tensor[:m]

        was_training = net.training
        net.eval()
        for cid in new_class_ids:
            xs = self._raw_tensors_for_class(dataset, cid)
            if xs is None:
                continue
            feats = []
            for i in range(0, xs.shape[0], batch_size):
                f = net.features(xs[i:i + batch_size].to(self.device))
                feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
            feats = torch.cat(feats, 0)
            chosen = _herding_indices(feats, m)
            self.by_class[cid] = xs[chosen].to(self.device)
        net.train(was_training)
        self.recompute_means(net, batch_size)

    @torch.no_grad()
    def recompute_means(self, net, batch_size: int = 64) -> None:
        was_training = net.training
        net.eval()
        for cid, xs in self.by_class.items():
            feats = []
            for i in range(0, xs.shape[0], batch_size):
                f = net.features(xs[i:i + batch_size].to(self.device))
                feats.append(torch.nn.functional.normalize(f, dim=1))
            mu = torch.cat(feats, 0).mean(0)
            self.means[cid] = torch.nn.functional.normalize(mu, dim=0)
        net.train(was_training)

    @torch.no_grad()
    def sample(self, batch_size: int):
        if self.is_empty:
            return None
        xs = torch.cat(list(self.by_class.values()), 0)
        ys = torch.cat([
            torch.full((v.shape[0],), cid, dtype=torch.long, device=self.device)
            for cid, v in self.by_class.items()
        ])
        idx = torch.from_numpy(
            np.random.choice(xs.shape[0], size=min(batch_size, xs.shape[0]), replace=False)
        ).to(self.device)
        return {"x": xs[idx], "y": ys[idx]}

    @torch.no_grad()
    def nme_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Nearest-mean-of-exemplars scores (negative distance), for prediction."""
        cids = sorted(self.means)
        mu = torch.stack([self.means[c] for c in cids], 0)
        f = torch.nn.functional.normalize(features, dim=1)
        return -torch.cdist(f, mu), cids

    @staticmethod
    def _raw_tensors_for_class(dataset, class_id: int):
        idx = [i for i, y in enumerate(dataset.targets) if y == class_id]
        if not idx:
            return None
        return torch.stack([dataset[i][2] for i in idx], 0)     # item 2 == x_raw

    def state(self) -> dict:
        return {"budget": self.budget, "filled": len(self), "classes": len(self.by_class)}


def _herding_indices(feats: torch.Tensor, m: int) -> list[int]:
    """Greedy selection so the exemplar mean tracks the true class mean."""
    m = min(m, feats.shape[0])
    mu = feats.mean(0)
    chosen: list[int] = []
    running = torch.zeros_like(mu)
    available = set(range(feats.shape[0]))
    for k in range(1, m + 1):
        cand = torch.tensor(sorted(available))
        scores = ((mu - (running + feats[cand]) / k) ** 2).sum(1)
        best = int(cand[int(scores.argmin())])
        chosen.append(best)
        available.discard(best)
        running = running + feats[best]
    return chosen
