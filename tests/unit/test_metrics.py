import numpy as np
import pytest

from effector_bincls.metrics import (
    find_optimal_threshold,
    high_recall_auprc,
    multi_scores,
)

PREDICTIONS = np.array([0.95, 0.82, 0.76, 0.61, 0.43, 0.31, 0.18, 0.07])
LABELS = np.array([1, 1, 1, 0, 1, 0, 0, 0])


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("youden", 0.76),
        ("f1", 0.31153846153846154),
        ("mcc", 0.31153846153846154),
    ],
)
def test_find_optimal_threshold_returns_expected_value(
    method: str,
    expected: float,
) -> None:
    assert find_optimal_threshold(PREDICTIONS, LABELS, method=method) == pytest.approx(
        expected
    )


def test_find_optimal_threshold_returns_expected_recall_constrained_value() -> None:
    assert find_optimal_threshold(
        PREDICTIONS,
        LABELS,
        method="recall_constrained",
        target_recall=0.75,
    ) == pytest.approx(0.6103893893893895)


def test_multi_scores_returns_expected_metrics() -> None:
    metrics = multi_scores(LABELS, PREDICTIONS, threshold=0.5)

    assert metrics == pytest.approx(
        {
            "roc_auc": 0.9375,
            "auprc": 0.95,
            "high_recall_auprc_0.7": 0.15625,
            "high_recall_auprc_0.8": 0.0,
            "accuracy": 0.75,
            "precision": 0.75,
            "recall": 0.75,
            "f1": 0.75,
            "mcc": 0.5,
            "TP": 3,
            "TN": 3,
            "FP": 1,
            "FN": 1,
        }
    )


def test_high_recall_auprc_returns_expected_value() -> None:
    assert high_recall_auprc(
        LABELS,
        PREDICTIONS,
        recall_threshold=0.7,
    ) == pytest.approx(0.15625)


def test_high_recall_auprc_returns_zero_when_threshold_unreachable() -> None:
    assert high_recall_auprc(
        LABELS,
        PREDICTIONS,
        recall_threshold=0.8,
    ) == pytest.approx(0.0)
