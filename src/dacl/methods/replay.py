"""Rehearsal strategies -- the strongest family when keeping imagery is allowed.

ER     -- replay stored examples with plain cross-entropy. Simple, and a much
          harder baseline than most papers admit.
DER++  -- also match the *logits* recorded when each example was stored, so the
          memory carries the old decision surface, not just labels. Usually the
          best accuracy/compute trade-off in our stream.
iCaRL  -- herding exemplars + distillation, and classify by nearest
          mean-of-exemplars in feature space. NME sidesteps the biased linear
          head entirely, which is why iCaRL degrades gracefully when the memory
          is small relative to the number of disaster classes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import ContinualMethod
from .buffer import HerdingMemory, ReservoirBuffer


class ER(ContinualMethod):
    """Params: buffer_size (500), replay_batch_size (= train batch size)."""

    name = "er"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.buffer = ReservoirBuffer(int(self.p("buffer_size", 500)), device)
        self.replay_bs = int(self.p("replay_batch_size", cfg.train.batch_size))

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        ce = self.ce(logits, y)
        rep = torch.zeros((), device=self.device)
        batch = self.buffer.sample(self.replay_bs)
        if batch is not None:
            rep = self.ce(self.net(batch["x"]), batch["y"])
        loss = ce + rep
        self.buffer.add(x_raw, y, task_id=self.task_index)
        return loss, {"ce": float(ce.detach()), "replay": float(rep.detach())}

    def extra_state(self) -> dict:
        return self.buffer.state()


class DERpp(ContinualMethod):
    """Params: buffer_size (500), alpha (0.5) logit-matching, beta (0.5) replay CE."""

    name = "derpp"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.buffer = ReservoirBuffer(int(self.p("buffer_size", 500)), device, store_logits=True)
        self.alpha = float(self.p("alpha", 0.5))
        self.beta = float(self.p("beta", 0.5))
        self.replay_bs = int(self.p("replay_batch_size", cfg.train.batch_size))

    def begin_task(self, task, scenario, train_loader) -> None:
        super().begin_task(task, scenario, train_loader)
        self.buffer.grow_logits(self.n_seen_classes)

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        ce = self.ce(logits, y)
        l_logit = torch.zeros((), device=self.device)
        l_ce = torch.zeros((), device=self.device)

        b1 = self.buffer.sample(self.replay_bs)
        if b1 is not None:
            out = self.net(b1["x"])
            stored = b1["logits"][:, : out.shape[1]]
            # Only the units that existed when the example was stored carry
            # signal; later columns are padding and must not be regressed to 0.
            width = min(stored.shape[1], out.shape[1])
            valid = (stored[:, :width].abs().sum(0) > 0)
            if valid.any():
                l_logit = F.mse_loss(out[:, :width][:, valid], stored[:, :width][:, valid])

        b2 = self.buffer.sample(self.replay_bs)
        if b2 is not None:
            l_ce = self.ce(self.net(b2["x"]), b2["y"])

        loss = ce + self.alpha * l_logit + self.beta * l_ce
        self.buffer.add(x_raw, y, logits=logits.detach(), task_id=self.task_index)
        return loss, {"ce": float(ce.detach()), "mse": float(l_logit.detach()),
                      "replay": float(l_ce.detach())}

    def extra_state(self) -> dict:
        return self.buffer.state()


class ICaRL(ContinualMethod):
    """Params: buffer_size (500) total exemplar budget, temperature (2.0),
    alpha (1.0) distillation weight, nme (True) use nearest-mean at test time."""

    name = "icarl"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.memory = HerdingMemory(int(self.p("buffer_size", 500)), device)
        self.temperature = float(self.p("temperature", 2.0))
        self.alpha = float(self.p("alpha", 1.0))
        self.use_nme = bool(self.p("nme", True))
        self.replay_bs = int(self.p("replay_batch_size", cfg.train.batch_size))
        self.teacher = None

    def observe(self, x, y, x_raw):
        if not self.memory.is_empty:
            mem = self.memory.sample(self.replay_bs)
            x = torch.cat([x, mem["x"]], 0)
            y = torch.cat([y, mem["y"]], 0)

        logits = self.net(x)
        ce = self.ce(logits, y)
        kd = torch.zeros((), device=self.device)
        if self.teacher is not None and self.n_old_classes > 0:
            with torch.no_grad():
                old = self.teacher(x)[:, : self.n_old_classes]
            kd = self.distillation(logits[:, : self.n_old_classes], old, self.temperature)
        loss = ce + self.alpha * kd
        return loss, {"ce": float(ce.detach()), "kd": float(kd.detach())}

    def end_task(self, task, scenario, train_loader) -> None:
        self.memory.update(self.net, task.train, task.class_ids)
        self.teacher = self.net.frozen_copy()

    @torch.no_grad()
    def predict(self, x) -> torch.Tensor:
        logits, feats = self.net(x, return_features=True)
        if not self.use_nme or self.memory.is_empty:
            return logits
        scores, cids = self.memory.nme_logits(feats)
        # Scatter NME scores back into full label space so the evaluator can
        # treat every method's output identically.
        out = torch.full_like(logits, float("-inf"))
        out[:, torch.tensor(cids, device=logits.device)] = scores.to(logits.dtype)
        return out

    def extra_state(self) -> dict:
        return self.memory.state()


class ERACE(ContinualMethod):
    """ER with asymmetric cross-entropy (Caccia et al., 2022).

    On the *incoming* batch, logits of classes that are absent from both the
    batch and the replay batch are masked out. This stops the sudden negative
    gradient that new data applies to old classes at a task boundary -- the
    mechanism behind most of the representation drift ER still suffers from.
    Params: buffer_size (500).
    """

    name = "er_ace"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        self.buffer = ReservoirBuffer(int(self.p("buffer_size", 500)), device)
        self.replay_bs = int(self.p("replay_batch_size", cfg.train.batch_size))
        self.seen_so_far = torch.zeros(0, dtype=torch.bool, device=device)

    def _mark_seen(self, y, n_classes):
        if self.seen_so_far.numel() < n_classes:
            grown = torch.zeros(n_classes, dtype=torch.bool, device=self.device)
            grown[: self.seen_so_far.numel()] = self.seen_so_far
            self.seen_so_far = grown
        self.seen_so_far[y.unique()] = True

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        n_classes = logits.shape[1]
        self._mark_seen(y, n_classes)

        mask = torch.zeros(n_classes, dtype=torch.bool, device=self.device)
        mask[y.unique()] = True
        batch = self.buffer.sample(self.replay_bs)
        if batch is not None:
            mask[batch["y"].unique()] = True
        else:
            mask |= ~self.seen_so_far[:n_classes]     # first task: no masking

        masked = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        ce = self.ce(masked, y)

        rep = torch.zeros((), device=self.device)
        if batch is not None:
            rep = self.ce(self.net(batch["x"]), batch["y"])

        loss = ce + rep
        self.buffer.add(x_raw, y, task_id=self.task_index)
        return loss, {"ce": float(ce.detach()), "replay": float(rep.detach())}

    def extra_state(self) -> dict:
        return self.buffer.state()
