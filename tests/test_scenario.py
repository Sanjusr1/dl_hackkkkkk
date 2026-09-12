"""Scenario construction is where a silent bug would invalidate every number:
a class leaking across tasks, or test data leaking into training, produces
plausible-looking results that are simply wrong."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dacl.config import Config
from dacl.data.datasets import (
    Sample,
    has_official_splits,
    partition_by_official_split,
    scan_folder,
    stratified_split,
)
from dacl.data.scenario import build_scenario


def _write_images(root: Path, layout: dict[tuple[str, ...], int], size: int = 16) -> None:
    rng = np.random.default_rng(0)
    for parts, n in layout.items():
        d = root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i:04d}.png")


@pytest.fixture
def flat_corpus(tmp_path):
    _write_images(tmp_path, {(c,): 12 for c in ["flood", "fire", "quake", "normal"]})
    return tmp_path


@pytest.fixture
def split_corpus(tmp_path):
    layout = {}
    for split, n in [("Train", 10), ("Validation", 4), ("Test", 6)]:
        for c in ["flood", "fire", "quake", "normal"]:
            layout[(split, c)] = n
    _write_images(tmp_path, layout)
    return tmp_path


def _cfg(root, **over):
    cfg = Config.from_dict({
        "seed": 0,
        "data": {"name": "folder", "root": str(root), "image_size": 16, "domain_from": "none"},
        "scenario": {"type": "class_incremental", "num_tasks": 3},
        "model": {"backbone": "smallcnn", "pretrained": False},
    })
    for path, value in over.items():
        cfg._set_path(path.split("."), value)
    return cfg


# ------------------------------------------------------------------- scanning
def test_scan_flat_layout(flat_corpus):
    samples = scan_folder(flat_corpus)
    assert len(samples) == 48
    assert {s.class_name for s in samples} == {"flood", "fire", "quake", "normal"}
    assert not has_official_splits(samples)


def test_scan_detects_official_splits(split_corpus):
    samples = scan_folder(split_corpus)
    assert has_official_splits(samples)
    assert {s.split for s in samples} == {"train", "val", "test"}


def test_official_split_folds_val_into_train(split_corpus):
    train, val, test = partition_by_official_split(scan_folder(split_corpus))
    assert len(train) == 4 * (10 + 4)
    assert val == []
    assert len(test) == 4 * 6


def test_domain_layout_detected(tmp_path):
    _write_images(tmp_path, {("event_a", "flood"): 5, ("event_b", "flood"): 5})
    samples = scan_folder(tmp_path)
    assert {s.domain for s in samples} == {"event_a", "event_b"}


def test_domain_from_none_collapses_domains(tmp_path):
    _write_images(tmp_path, {("event_a", "flood"): 5, ("event_b", "flood"): 5})
    samples = scan_folder(tmp_path, domain_from="none")
    assert {s.domain for s in samples} == {"default"}


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path / "nope")


# ------------------------------------------------------------------ splitting
def test_stratified_split_is_disjoint_and_deterministic(flat_corpus):
    samples = scan_folder(flat_corpus)
    a = stratified_split(samples, 0.25, seed=0)
    b = stratified_split(samples, 0.25, seed=0)
    assert [s.path for s in a[0]] == [s.path for s in b[0]]
    train_paths = {s.path for s in a[0]}
    test_paths = {s.path for s in a[2]}
    assert train_paths.isdisjoint(test_paths)
    assert len(test_paths) == 12          # 25% of 12 per class, 4 classes


def test_stratified_split_keeps_every_class_in_test(flat_corpus):
    samples = scan_folder(flat_corpus)
    _, _, test = stratified_split(samples, 0.25, seed=1)
    assert {s.class_name for s in test} == {"flood", "fire", "quake", "normal"}


# ------------------------------------------------------------------ scenarios
def test_cil_chunking_is_2_1_1_for_four_classes(flat_corpus):
    scenario = build_scenario(_cfg(flat_corpus))
    assert [t.num_classes for t in scenario] == [2, 1, 1]


def test_cil_classes_never_overlap(flat_corpus):
    scenario = build_scenario(_cfg(flat_corpus))
    seen: set[str] = set()
    for task in scenario:
        assert not (seen & set(task.class_names))
        seen |= set(task.class_names)
    assert seen == set(scenario.classes)


def test_cil_global_ids_are_contiguous_per_task(flat_corpus):
    """Task t must own a contiguous id block, or the growing head misaligns."""
    scenario = build_scenario(_cfg(flat_corpus))
    expected = 0
    for task in scenario:
        assert task.class_ids == list(range(expected, expected + task.num_classes))
        expected += task.num_classes


def test_classes_seen_after_grows(flat_corpus):
    scenario = build_scenario(_cfg(flat_corpus))
    assert [scenario.classes_seen_after(i) for i in range(3)] == [2, 3, 4]


def test_class_order_follows_seed(flat_corpus):
    a = build_scenario(_cfg(flat_corpus, seed=0)).classes
    orders = {tuple(build_scenario(_cfg(flat_corpus, seed=s)).classes) for s in range(6)}
    assert len(orders) > 1, "class order should vary with the seed"
    assert tuple(build_scenario(_cfg(flat_corpus, seed=0)).classes) == tuple(a)


def test_explicit_task_grouping_respected(flat_corpus):
    cfg = _cfg(flat_corpus)
    cfg.scenario.tasks = [["normal", "flood"], ["fire"], ["quake"]]
    scenario = build_scenario(cfg)
    assert [t.class_names for t in scenario] == [["normal", "flood"], ["fire"], ["quake"]]
    assert scenario.classes == ["normal", "flood", "fire", "quake"]


def test_unknown_class_in_grouping_raises(flat_corpus):
    cfg = _cfg(flat_corpus)
    cfg.scenario.tasks = [["normal", "volcano"], ["fire"]]
    with pytest.raises(ValueError):
        build_scenario(cfg)


def test_duplicate_class_across_tasks_raises(flat_corpus):
    cfg = _cfg(flat_corpus)
    cfg.scenario.tasks = [["normal", "flood"], ["flood"]]
    with pytest.raises(ValueError):
        build_scenario(cfg)


def test_official_splits_used_by_scenario(split_corpus):
    scenario = build_scenario(_cfg(split_corpus))
    assert sum(len(t.train) for t in scenario) == 4 * 14
    assert sum(len(t.test) for t in scenario) == 4 * 6


def test_dil_shares_one_label_space(tmp_path):
    layout = {}
    for dom in ["event_a", "event_b"]:
        for c in ["flood", "fire"]:
            layout[(dom, c)] = 8
    _write_images(tmp_path, layout)
    cfg = _cfg(tmp_path)
    cfg.data.domain_from = "parent"
    cfg.scenario.type = "domain_incremental"
    cfg.scenario.num_tasks = 2
    scenario = build_scenario(cfg)
    assert scenario.num_classes == 2
    assert all(scenario.classes_seen_after(i) == 2 for i in range(2))


def test_dil_requires_multiple_domains(flat_corpus):
    cfg = _cfg(flat_corpus)
    cfg.scenario.type = "domain_incremental"
    with pytest.raises(ValueError):
        build_scenario(cfg)


# --------------------------------------------------------------------- config
def test_config_base_chain_resolves():
    cfg = Config.load(Path(__file__).parents[1] / "configs" / "aiderv2_cil_fast.yaml")
    assert cfg.model.backbone == "resnet18"       # from base.yaml via aiderv2_cil
    assert cfg.data.image_size == 128             # overridden by the fast config
    assert cfg.train.epochs == 6


def test_config_overrides_nested_params():
    cfg = Config.from_dict({"method": {"name": "derpp", "params": {"alpha": 0.5}}})
    cfg.apply_overrides(["method.params.alpha=0.9", "train.epochs=3"])
    assert cfg.method.params["alpha"] == 0.9
    assert cfg.train.epochs == 3


def test_unknown_config_key_raises():
    with pytest.raises(KeyError):
        Config.from_dict({"train": {"not_a_field": 1}})
