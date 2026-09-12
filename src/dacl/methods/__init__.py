"""Registry: `method.name` in the config resolves to a class here."""
from __future__ import annotations

from .base import ContinualMethod
from .baselines import Finetune, FrozenFeatures, Joint
from .regularization import EWC, LwF
from .replay import ER, ERACE, DERpp, ICaRL

REGISTRY: dict[str, type[ContinualMethod]] = {
    m.name: m for m in [Finetune, Joint, FrozenFeatures, EWC, LwF, ER, ERACE, DERpp, ICaRL]
}


def build_method(cfg, net, device) -> ContinualMethod:
    name = cfg.method.name.lower()
    if name not in REGISTRY:
        raise KeyError(f"Unknown method {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](net, cfg, device)


__all__ = ["REGISTRY", "build_method", "ContinualMethod"]
