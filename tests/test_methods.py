"""Buffer, head-growth and per-method integration tests.

The integration test runs every registered strategy over a real 2-task stream.
It does not assert accuracy -- it asserts the thing that actually breaks when a
method is added: shape mismatches as the label space grows, non-finite losses,
and memories that quietly stay empty.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from dacl.config import Config
from dacl.data.scenario import build_scenario
from dacl.engine.trainer import ContinualTrainer
from dacl.methods import REGISTRY, build_method
from dacl.methods.buffer import HerdingMemory, ReservoirBuffer, _herding_indices
from dacl.models.network import DACLNet
from dacl.utils import setup_logging

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------- buffers
def test_reservoir_respects_capacity():
    buf = ReservoirBuffer(10, DEVICE)
    for _ in range(20):
        buf.add(torch.randn(4, 3, 8, 8), torch.randint(0, 3, (4,)))
    assert len(buf) == 10
    assert buf.n_seen == 80


def test_reservoir_sample_shapes():
    buf = ReservoirBuffer(16, DEVICE)
    buf.add(torch.randn(8, 3, 8, 8), torch.randint(0, 3, (8,)))
    batch = buf.sample(4)
    assert batch["x"].shape == (4, 3, 8, 8)
    assert batch["y"].shape == (4,)


def test_empty_reservoir_returns_none():
    assert ReservoirBuffer(10, DEVICE).sample(4) is None
    assert ReservoirBuffer(0, DEVICE).sample(4) is None


def test_reservoir_is_approximately_uniform():
    """Each seen item should survive with probability capacity/n_seen."""
    torch.manual_seed(0)
    np.random.seed(0)
    counts = np.zeros(100)
    trials = 60
    for _ in range(trials):
        buf = ReservoirBuffer(10, DEVICE)
        for i in range(100):
            buf.add(torch.zeros(1, 3, 4, 4), torch.tensor([i]))
        counts[buf.y.cpu().numpy()] += 1
    early, late = counts[:50].mean(), counts[50:].mean()
    assert abs(early - late) < 0.35 * trials, "reservoir is badly biased by arrival order"


def test_grow_logits_pads_without_losing_data():
    buf = ReservoirBuffer(8, DEVICE, store_logits=True)
    buf.add(torch.randn(4, 3, 8, 8), torch.randint(0, 2, (4,)), logits=torch.randn(4, 2))
    before = buf.logits[:, :2].clone()
    buf.grow_logits(5)
    assert buf.logits.shape[1] == 5
    assert torch.allclose(buf.logits[:, :2], before)
    assert torch.all(buf.logits[:, 2:] == 0)


def test_herding_picks_requested_count_and_no_duplicates():
    feats = torch.randn(30, 8)
    idx = _herding_indices(feats, 7)
    assert len(idx) == 7
    assert len(set(idx)) == 7


def test_herding_mean_beats_random_mean():
    """Herding exists to track the class mean better than random selection."""
    torch.manual_seed(0)
    feats = torch.randn(200, 16)
    mu = feats.mean(0)
    chosen = _herding_indices(feats, 10)
    herd_err = (feats[chosen].mean(0) - mu).norm()
    rand_errs = [
        (feats[torch.randperm(200)[:10]].mean(0) - mu).norm() for _ in range(20)
    ]
    assert herd_err < np.median([float(e) for e in rand_errs])


def test_herding_memory_quota_shrinks_as_classes_grow():
    mem = HerdingMemory(100, DEVICE)
    assert mem.quota(4) == 25
    assert mem.quota(10) == 10


# ----------------------------------------------------------------------- model
def test_head_grows_and_ids_stay_aligned():
    net = DACLNet(backbone="smallcnn", pretrained=False, head="cosine")
    net.expand(2, 0)
    net.expand(3, 1)
    assert net.num_classes == 5
    assert net.class_to_task == [0, 0, 1, 1, 1]
    out = net(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 5)


def test_cosine_head_logits_are_bounded_by_scale():
    net = DACLNet(backbone="smallcnn", pretrained=False, head="cosine")
    net.expand(4, 0)
    out = net(torch.randn(8, 3, 32, 32))
    scale = float(net.head.scale.detach())
    assert out.abs().max() <= scale + 1e-4


def test_weight_align_rescales_new_classes():
    net = DACLNet(backbone="smallcnn", pretrained=False, head="linear")
    net.expand(2, 0)
    with torch.no_grad():
        net.head.heads[0].weight.mul_(0.1)      # old classes: small norms
    net.expand(2, 1)
    before = float(net.head.heads[-1].weight.detach().norm(dim=1).mean())
    net.head.weight_align(2)
    after = float(net.head.heads[-1].weight.detach().norm(dim=1).mean())
    old = float(net.head.heads[0].weight.detach().norm(dim=1).mean())
    assert after < before
    assert after == pytest.approx(old, rel=1e-4)


def test_frozen_copy_is_detached():
    net = DACLNet(backbone="smallcnn", pretrained=False)
    net.expand(2, 0)
    teacher = net.frozen_copy()
    assert not any(p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    x = torch.randn(2, 3, 32, 32)
    assert not torch.allclose(net(x), teacher(x)), "teacher must not track the student"


# ----------------------------------------------------------------- integration
@pytest.fixture
def tiny_stream(tmp_path):
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(["a", "b", "c", "d"]):
        d = tmp_path / cls
        d.mkdir(parents=True)
        for i in range(8):
            # class-correlated colour so the task is learnable at all
            arr = np.clip(rng.normal(60 * ci + 40, 12, (16, 16, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")
    cfg = Config.from_dict({
        "seed": 0,
        "eval_future_tasks": False,
        "data": {"name": "folder", "root": str(tmp_path), "image_size": 16,
                 "domain_from": "none", "test_fraction": 0.25},
        "scenario": {"type": "class_incremental", "num_tasks": 2, "shuffle_classes": False},
        "model": {"backbone": "smallcnn", "pretrained": False, "head": "cosine"},
        "train": {"epochs": 1, "batch_size": 4, "eval_batch_size": 4,
                  "backbone_lr_scale": 1.0, "scheduler": "none"},
        "method": {"name": "finetune", "params": {"buffer_size": 16, "fisher_batches": 2}},
    })
    return cfg


@pytest.mark.parametrize("method_name", sorted(REGISTRY))
def test_every_method_completes_a_stream(tiny_stream, method_name):
    cfg = tiny_stream
    cfg.method.name = method_name
    scenario = build_scenario(cfg)
    trainer = ContinualTrainer(cfg, scenario, DEVICE, setup_logging(None))
    results = trainer.run()

    R = np.array(results["accuracy_matrix"], dtype=float)
    assert R.shape == (2, 2)
    seen = R[np.tril_indices(2)]
    assert np.all(np.isfinite(seen)), f"{method_name} left an unevaluated seen task"
    assert np.all((seen >= 0) & (seen <= 1))
    for key in ["final_average_accuracy", "backward_transfer", "forgetting"]:
        assert np.isfinite(results["metrics"][key])


def test_losses_are_finite_for_every_method(tiny_stream):
    for name in sorted(REGISTRY):
        cfg = tiny_stream
        cfg.method.name = name
        scenario = build_scenario(cfg)
        net = DACLNet(backbone="smallcnn", pretrained=False, head="cosine")
        net.expand(scenario.tasks[0].num_classes, 0)
        method = build_method(cfg, net, DEVICE)
        method.begin_task(scenario.tasks[0], scenario, None)
        x = torch.randn(4, 3, 16, 16)
        y = torch.randint(0, scenario.tasks[0].num_classes, (4,))
        loss, parts = method.observe(x, y, x)
        assert torch.isfinite(loss), f"{name} produced a non-finite loss"
        loss.backward()


def test_replay_methods_actually_fill_memory(tiny_stream):
    for name in ["er", "er_ace", "derpp", "icarl"]:
        cfg = tiny_stream
        cfg.method.name = name
        scenario = build_scenario(cfg)
        trainer = ContinualTrainer(cfg, scenario, DEVICE, setup_logging(None))
        trainer.run()
        state = trainer.method.extra_state()
        assert state.get("filled", 0) > 0, f"{name} finished with an empty memory"


def test_joint_sees_cumulative_data(tiny_stream):
    cfg = tiny_stream
    cfg.method.name = "joint"
    scenario = build_scenario(cfg)
    trainer = ContinualTrainer(cfg, scenario, DEVICE, setup_logging(None))
    assert trainer.method.wants_cumulative_data
    trainer.run()
