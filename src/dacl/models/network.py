"""Backbone + growing classifier.

Design decisions that matter for continual learning:

1. The head *grows*. Each task appends its own output units instead of
   re-allocating one giant layer, so old units keep their identity and their
   optimiser state can be handled explicitly.

2. A cosine head is the default. With a plain linear head, the classes of the
   most recent task get systematically larger logit norms simply because they
   are the only ones with abundant gradient -- "task-recency bias", the single
   biggest cause of class-incremental error beyond feature drift. Normalising
   features and weights removes the norm degree of freedom.

3. `weight_align()` implements the WA correction (Zhao et al., 2020) for linear
   heads: rescale new-class weights so their mean norm matches old classes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- backbones
class SmallCNN(nn.Module):
    """Tiny conv net for CPU debugging and unit tests (~0.3M params)."""

    def __init__(self, width: int = 32, dropout: float = 0.0):
        super().__init__()
        c = width
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, c), block(c, 2 * c), block(2 * c, 4 * c))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 4 * c

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.drop(h)


def build_backbone(name: str, pretrained: bool, dropout: float = 0.0) -> tuple[nn.Module, int]:
    if name == "smallcnn":
        net = SmallCNN(dropout=dropout)
        return net, net.out_dim

    import torchvision.models as tvm

    factories = {
        "resnet18": (tvm.resnet18, "ResNet18_Weights"),
        "resnet34": (tvm.resnet34, "ResNet34_Weights"),
        "resnet50": (tvm.resnet50, "ResNet50_Weights"),
    }
    if name not in factories:
        raise ValueError(f"Unknown backbone {name!r}")
    fn, weights_enum = factories[name]

    weights = None
    if pretrained:
        try:
            weights = getattr(tvm, weights_enum).DEFAULT
        except Exception:
            weights = None
    try:
        net = fn(weights=weights)
    except Exception as exc:   # offline machine, no cached weights
        print(f"[warn] pretrained weights unavailable ({exc}); using random init")
        net = fn(weights=None)

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    if dropout > 0:
        net = nn.Sequential(net, nn.Dropout(dropout))
    return net, feat_dim


# ------------------------------------------------------------------------ heads
class IncrementalLinearHead(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.heads = nn.ModuleList()

    @property
    def out_features(self) -> int:
        return sum(h.out_features for h in self.heads)

    def add_classes(self, n: int) -> None:
        if n <= 0:
            return
        layer = nn.Linear(self.feat_dim, n)
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
        nn.init.zeros_(layer.bias)
        self.heads.append(layer)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(f) for h in self.heads], dim=1)

    @torch.no_grad()
    def weight_align(self, n_new: int) -> None:
        """Rescale the newest classes so their weight norms match the old ones."""
        if len(self.heads) < 2 or n_new <= 0:
            return
        w_old = torch.cat([h.weight for h in self.heads[:-1]], 0)
        w_new = self.heads[-1].weight
        gamma = w_old.norm(dim=1).mean() / (w_new.norm(dim=1).mean() + 1e-8)
        self.heads[-1].weight.mul_(gamma)
        self.heads[-1].bias.mul_(gamma)


class IncrementalCosineHead(nn.Module):
    """Normalised features x normalised prototypes, with a learnable scale."""

    def __init__(self, feat_dim: int, scale: float = 16.0, learn_scale: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.weights = nn.ParameterList()
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=learn_scale)

    @property
    def out_features(self) -> int:
        return sum(w.shape[0] for w in self.weights)

    def add_classes(self, n: int) -> None:
        if n <= 0:
            return
        w = nn.Parameter(torch.empty(n, self.feat_dim))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weights.append(w)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        w = torch.cat(list(self.weights), 0)
        return self.scale * F.linear(F.normalize(f, dim=1), F.normalize(w, dim=1))

    @torch.no_grad()
    def weight_align(self, n_new: int) -> None:
        return  # cosine head is norm-invariant by construction


# ------------------------------------------------------------------ full model
class DACLNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 head: str = "cosine", dropout: float = 0.0, freeze_backbone: bool = False):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained, dropout)
        self.head_type = head
        self.head = (IncrementalCosineHead(self.feat_dim) if head == "cosine"
                     else IncrementalLinearHead(self.feat_dim))
        self.class_to_task: list[int] = []
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ growth
    def expand(self, n_new_classes: int, task_index: int) -> None:
        self.head.add_classes(n_new_classes)
        self.class_to_task += [task_index] * n_new_classes

    @property
    def num_classes(self) -> int:
        return self.head.out_features

    # ----------------------------------------------------------------- forward
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.features(x)
        logits = self.head(f)
        if return_features:
            return logits, f
        return logits

    # ------------------------------------------------------------- convenience
    def param_groups(self, lr: float, backbone_lr_scale: float = 1.0):
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr * backbone_lr_scale},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr": lr},
        ]

    def frozen_copy(self) -> "DACLNet":
        """Deep copy in eval mode with grads off -- the teacher for LwF/iCaRL."""
        import copy

        clone = copy.deepcopy(self)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone
