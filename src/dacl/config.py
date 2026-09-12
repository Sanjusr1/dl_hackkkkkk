"""Typed experiment configuration.

Everything an experiment needs is declared here so a run is fully described by
one YAML file plus a seed. `--set a.b=c` overrides let us sweep without editing
files.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    name: str = "synthetic"          # synthetic | folder | xbd_patches
    root: str = "data/synthetic"
    image_size: int = 64
    test_fraction: float = 0.2       # used when the dataset has no official split
    val_fraction: float = 0.0
    use_official_splits: bool = True  # honour train/val/test dirs when present
    num_workers: int = 0
    classes: list[str] | None = None  # restrict / order the label space
    domain_from: str = "parent"      # for domain-incremental: where the event id lives
    synthetic_per_class: int = 120
    max_per_class: int | None = None  # subsample for quick debugging


@dataclass
class ScenarioConfig:
    type: str = "class_incremental"   # class_incremental | domain_incremental
    num_tasks: int = 3
    tasks: list[list[str]] | None = None   # explicit grouping; overrides num_tasks
    class_order_seed: int | None = None    # None -> use the global seed
    shuffle_classes: bool = True


@dataclass
class ModelConfig:
    backbone: str = "resnet18"       # smallcnn | resnet18 | resnet34 | resnet50
    pretrained: bool = True
    head: str = "cosine"             # linear | cosine
    dropout: float = 0.0
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 1e-3
    backbone_lr_scale: float = 0.1   # smaller LR on pretrained features = less drift
    weight_decay: float = 5e-4
    optimizer: str = "adam"          # adam | sgd
    momentum: float = 0.9
    scheduler: str = "cosine"        # cosine | step | none
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    reset_optimizer_each_task: bool = True


@dataclass
class MethodConfig:
    name: str = "derpp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/debug"
    tag: str = ""
    eval_future_tasks: bool = True   # needed for forward-transfer
    data: DataConfig = field(default_factory=DataConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    method: MethodConfig = field(default_factory=MethodConfig)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = dict(raw or {})
        sub = {
            "data": DataConfig,
            "scenario": ScenarioConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "method": MethodConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            kwargs[key] = _build(klass, raw.pop(key, {}) or {})
        for key in list(raw):
            if key not in {f.name for f in dataclasses.fields(cls)}:
                raise KeyError(f"Unknown config key: {key}")
            kwargs[key] = raw[key]
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(_load_raw(Path(path)))

    def apply_overrides(self, overrides: list[str]) -> "Config":
        """Apply `a.b=value` strings coming from the command line."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must look like key=value, got {item!r}")
            key, value = item.split("=", 1)
            self._set_path(key.split("."), _parse_scalar(value))
        return self

    def _set_path(self, parts: list[str], value: Any) -> None:
        node: Any = self
        for p in parts[:-1]:
            if isinstance(node, dict):
                node = node.setdefault(p, {})
            else:
                node = getattr(node, p)
        leaf = parts[-1]
        if isinstance(node, dict):
            node[leaf] = value
        else:
            if not hasattr(node, leaf):
                raise KeyError(f"Unknown config field: {'.'.join(parts)}")
            setattr(node, leaf, value)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def run_name(self) -> str:
        bits = [self.data.name, self.scenario.type[:3], self.method.name, f"s{self.seed}"]
        if self.tag:
            bits.append(self.tag)
        return "-".join(bits)


def _build(klass, raw: dict[str, Any]):
    known = {f.name for f in dataclasses.fields(klass)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"Unknown keys for {klass.__name__}: {sorted(unknown)}")
    return klass(**raw)


def _load_raw(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Read a config, resolving `_base_` chains of arbitrary depth."""
    import yaml

    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular _base_ chain at {path}")
    _seen.add(path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base_", None)
    if base is None:
        return raw
    return _deep_merge(_load_raw(path.parent / base, _seen), raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(value: str) -> Any:
    import yaml

    try:
        return yaml.safe_load(value)
    except Exception:
        return value
