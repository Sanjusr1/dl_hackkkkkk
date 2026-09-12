"""The two reference points every continual number must be read against.

Finetune -- train each task with nothing to protect the past. Its gap to Joint
            *is* catastrophic forgetting; a method that does not beat it is not
            doing anything.
Joint    -- retrain on all data seen so far at every step. Not a continual
            method (it needs unbounded storage and unbounded compute) but it
            bounds what any method could achieve on this stream.
"""
from __future__ import annotations

import torch

from .base import ContinualMethod


class Finetune(ContinualMethod):
    name = "finetune"

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        loss = self.ce(logits, y)
        return loss, {"ce": float(loss.detach())}


class Joint(ContinualMethod):
    """Identical loss to Finetune; the trainer feeds it cumulative data."""

    name = "joint"
    wants_cumulative_data = True

    def observe(self, x, y, x_raw):
        logits = self.net(x)
        loss = self.ce(logits, y)
        return loss, {"ce": float(loss.detach())}


class FrozenFeatures(ContinualMethod):
    """Cheap, surprisingly strong control: freeze pretrained features, grow only
    the head. Bounds how much of any method's score comes from the ImageNet
    initialisation rather than from continual learning itself."""

    name = "frozen"

    def __init__(self, net, cfg, device):
        super().__init__(net, cfg, device)
        for p in self.net.backbone.parameters():
            p.requires_grad = False

    def observe(self, x, y, x_raw):
        with torch.no_grad():
            f = self.net.features(x)
        logits = self.net.head(f)
        loss = self.ce(logits, y)
        return loss, {"ce": float(loss.detach())}
