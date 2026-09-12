"""Metrics are the thing every claim in the report rests on, so they are
checked against hand-computed values rather than against themselves."""
import numpy as np
import pytest

from dacl.engine.metrics import (
    average_incremental_accuracy,
    backward_transfer,
    final_average_accuracy,
    forgetting,
    forward_transfer,
    learning_accuracy,
    macro_f1,
    per_class_recall,
    summarize,
)


@pytest.fixture
def R_forgetting():
    """Total forgetting: each task perfect when learned, zero afterwards."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def R_perfect():
    return np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])


def test_total_forgetting(R_forgetting):
    assert final_average_accuracy(R_forgetting) == pytest.approx(1 / 3)
    assert backward_transfer(R_forgetting) == pytest.approx(-1.0)
    assert forgetting(R_forgetting) == pytest.approx(1.0)
    assert learning_accuracy(R_forgetting) == pytest.approx(1.0)


def test_no_forgetting(R_perfect):
    assert final_average_accuracy(R_perfect) == pytest.approx(1.0)
    assert backward_transfer(R_perfect) == pytest.approx(0.0)
    assert forgetting(R_perfect) == pytest.approx(0.0)


def test_average_incremental_accuracy(R_perfect):
    # mean over steps of the mean over tasks seen so far -> 1.0 when perfect
    assert average_incremental_accuracy(R_perfect) == pytest.approx(1.0)


def test_average_incremental_accuracy_partial(R_forgetting):
    # steps: 1.0, mean(0,1)=0.5, mean(0,0,1)=1/3
    expected = (1.0 + 0.5 + 1 / 3) / 3
    assert average_incremental_accuracy(R_forgetting) == pytest.approx(expected)


def test_learning_accuracy_separates_plasticity():
    """A rigid method that never learns new tasks has low LA but no forgetting.

    This is the EWC-with-huge-lambda failure mode, and average accuracy alone
    cannot distinguish it from a method that learns and then forgets.
    """
    rigid = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [1.0, 0.2, 0.2],
    ])
    assert learning_accuracy(rigid) == pytest.approx((1.0 + 0.2 + 0.2) / 3)
    assert backward_transfer(rigid) == pytest.approx(0.0)


def test_forgetting_penalises_transient_collapse():
    """Task 0 peaks at 1.0, collapses, partially recovers: BWT understates it."""
    R = np.array([
        [1.0, np.nan, np.nan],
        [0.1, 1.0, np.nan],
        [0.8, 1.0, 1.0],
    ])
    assert backward_transfer(R) == pytest.approx((0.8 - 1.0 + 1.0 - 1.0) / 2)
    # best-ever for task 0 over rows 0..1 is 1.0 -> drop of 0.2
    assert forgetting(R) == pytest.approx((1.0 - 0.8 + 1.0 - 1.0) / 2)


def test_forward_transfer_vs_random_baseline():
    R = np.array([
        [1.0, 0.6, 0.5],
        [0.0, 1.0, 0.7],
        [0.0, 0.0, 1.0],
    ])
    baseline = np.array([0.3, 0.3, 0.3])
    # uses R[0,1]-b1 and R[1,2]-b2
    assert forward_transfer(R, baseline) == pytest.approx(((0.6 - 0.3) + (0.7 - 0.3)) / 2)


def test_forward_transfer_nan_without_baseline():
    R = np.eye(3)
    assert np.isnan(forward_transfer(R, None))


def test_single_task_stream_is_degenerate():
    R = np.array([[0.9]])
    assert backward_transfer(R) == 0.0
    assert forgetting(R) == 0.0
    assert final_average_accuracy(R) == pytest.approx(0.9)


def test_summarize_keys():
    out = summarize(np.eye(3), np.zeros(3))
    for key in ["final_average_accuracy", "backward_transfer", "forgetting",
                "learning_accuracy", "average_incremental_accuracy"]:
        assert key in out


def test_macro_f1_ignores_absent_classes():
    conf = np.array([[5, 0, 0],
                     [0, 5, 0],
                     [0, 0, 0]])      # class 2 has no support
    assert macro_f1(conf) == pytest.approx(1.0)


def test_macro_f1_penalises_collapsed_class():
    """Accuracy stays high when a rare class is dropped; macro-F1 must not."""
    conf = np.array([[90, 0],
                     [10, 0]])        # class 1 never predicted
    accuracy = np.trace(conf) / conf.sum()
    assert accuracy == pytest.approx(0.9)
    assert macro_f1(conf) < 0.5


def test_per_class_recall_nan_for_no_support():
    conf = np.array([[4, 1],
                     [0, 0]])
    rec = per_class_recall(conf)
    assert rec[0] == pytest.approx(0.8)
    assert np.isnan(rec[1])
