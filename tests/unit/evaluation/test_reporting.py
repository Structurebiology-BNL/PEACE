import numpy as np

from effector_bincls.evaluation.reporting import _find_roc_marker_index


def test_find_roc_marker_index_skips_non_finite_thresholds() -> None:
    thresholds = np.array([np.inf, 0.9, 0.41, 0.1])

    assert _find_roc_marker_index(thresholds, 0.4) == 2


def test_find_roc_marker_index_returns_none_when_no_finite_thresholds() -> None:
    thresholds = np.array([np.inf, np.inf])

    assert _find_roc_marker_index(thresholds, 0.4) is None
