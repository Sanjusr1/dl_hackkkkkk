"""Rehearsal-free strategies -- the only option when imagery cannot be retained.

This matters operationally: satellite products supplied under a Charter
activation often come with redistribution restrictions, and a partner may be
allowed to train on imagery but not to keep it. EWC and LwF both fit that
constraint, at the cost of being weaker than replay.

EWC  (Kirkpatrick et al., 2017) -- penalise movement of weights that mattered
     to previous tasks, importance = diagonal of the empirical Fisher.
LwF  (Li & Hoiem, 2016)          -- keep the *function* stable instead of the
     weights: distil the frozen previous model's old-class logits on new data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import ContinualMethod


class EWC(ContinualMethod):
    """Params: lambda_ewc (default 5000), fisher_batches (50), online (True), gamma (0.95)."""

    name = "ewc"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.lam = float(self.p("lambda_ewc", 5000.0))
        self.online = bool(self.p("online", True))
        self.gamma = float(self.p("gamma", 0.95))
        self.fisher_batches = int(self.p("fisher_batches", 50))
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        params = dict(self.net.named_parameters())
        for name, fisher in self.fisher.items():
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue          # head units added after the anchor was taken
            total = total + (fisher * (p - self.anchor[name]) ** 2).sum()
        return 0.5 * total

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        ce = self.ce(logits, y)
        reg = self.penalty()
        loss = ce + self.lam * reg
        return loss, {"ce": float(ce.detach()), "ewc": float(reg.detach())}

    @torch.no_grad()
    def _snapshot(self) -> None:
        self.anchor = {
            n: p.detach().clone()
            for n, p in self.net.named_parameters()
            if p.requires_grad and not n.startswith("head")
        }

    def end_task(self, task, scenario, train_loader) -> None:
        """Empirical diagonal Fisher on this task's data, at the task optimum."""
        new_fisher = {
            n: torch.zeros_like(p)
            for n, p in self.net.named_parameters()
            if p.requires_grad and not n.startswith("head")
        }
        self.net.eval()
        seen = 0
        for b, (x, y, _) in enumerate(train_loader):
            if b >= self.fisher_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.net.zero_grad(set_to_none=True)
            logits = self.net(x)
            # Empirical Fisher: gradient of the log-likelihood of the true label.
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in self.net.named_parameters():
                if n in new_fisher and p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2 * x.size(0)
            seen += x.size(0)
        self.net.zero_grad(set_to_none=True)
        self.net.train()
        if seen == 0:
            return
        for n in new_fisher:
            new_fisher[n] /= seen

        if self.online and self.fisher:
            # One consolidated Fisher, decayed -- O(1) memory in the task count.
            for n, f in new_fisher.items():
                self.fisher[n] = self.gamma * self.fisher.get(n, torch.zeros_like(f)) + f
        else:
            for n, f in new_fisher.items():
                self.fisher[n] = self.fisher.get(n, torch.zeros_like(f)) + f
        self._snapshot()

    def extra_state(self) -> dict:
        n = sum(v.numel() for v in self.fisher.values())
        return {"fisher_params": n, "online": self.online}


class LwF(ContinualMethod):
    """Params: alpha (1.0), temperature (2.0)."""

    name = "lwf"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.alpha = float(self.p("alpha", 1.0))
        self.temperature = float(self.p("temperature", 2.0))
        self.teacher = None

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        ce = self.ce(logits, y)
        kd = torch.zeros((), device=self.device)
        if self.teacher is not None and self.n_old_classes > 0:
            with torch.no_grad():
                old_logits = self.teacher(x)[:, : self.n_old_classes]
            kd = self.distillation(logits[:, : self.n_old_classes], old_logits, self.temperature)
        # Scaling by the fraction of old classes keeps the trade-off stable as
        # the label space grows -- without it, KD is under-weighted late in the
        # stream, exactly when there is most to forget.
        scale = self.n_old_classes / max(1, self.n_seen_classes)
        loss = ce + self.alpha * scale * kd
        return loss, {"ce": float(ce.detach()), "kd": float(kd.detach())}

    def end_task(self, task, scenario, train_loader) -> None:
        self.teacher = self.net.frozen_copy()

    def extra_state(self) -> dict:
        return {"teacher": self.teacher is not None}
