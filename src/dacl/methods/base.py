"""Common interface every continual-learning strategy implements.

The trainer owns the loop and the optimiser; a method only decides
  * what happens before a task starts   (`begin_task`)
  * what the loss on a minibatch is      (`observe`)
  * what happens after a task ends       (`end_task`)
  * optionally, how predictions are made (`predict`)

Keeping the contract this small is what lets us swap eight strategies through
one config key and keep the comparison honest -- identical data order, identical
optimiser, identical augmentation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinualMethod:
    name = "base"
    wants_cumulative_data = False      # True only for the Joint upper bound

    def __init__(self, net, cfg, device):
        self.net = net
        self.cfg = cfg
        self.device = device
        self.params: dict[str, Any] = dict(cfg.method.params or {})
        self.task_index = -1
        self.n_seen_classes = 0
        self.n_old_classes = 0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    # ------------------------------------------------------------------ hooks
    def begin_task(self, task, scenario, train_loader) -> None:
        self.task_index = task.index
        self.n_old_classes = self.n_seen_classes
        self.n_seen_classes = scenario.classes_seen_after(task.index)

    def observe(self, x, y, x_raw) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def end_task(self, task, scenario, train_loader) -> None:
        pass

    @torch.no_grad()
    def predict(self, x) -> torch.Tensor:
        return self.net(x)

    # --------------------------------------------------------------- utilities
    def p(self, key: str, default):
        """Read a method hyper-parameter with a documented default."""
        return self.params.get(key, default)

    def ce(self, logits, y) -> torch.Tensor:
        return self.criterion(logits, y)

    @staticmethod
    def distillation(student_logits, teacher_logits, temperature: float = 2.0) -> torch.Tensor:
        """Standard KD term (Hinton), restricted by the caller to old classes."""
        if teacher_logits.numel() == 0:
            return student_logits.new_zeros(())
        t = temperature
        loss = F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.log_softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
            log_target=True,
        )
        return loss * (t ** 2)

    def extra_state(self) -> dict:
        """Anything worth logging about the method's memory footprint."""
        return {}
