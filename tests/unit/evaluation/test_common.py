from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from effector_bincls.evaluation.common import (
    collect_oof_predictions,
    find_global_optimal_threshold,
    parse_evaluation_args,
)
from effector_bincls.metrics import find_optimal_threshold


def test_parse_evaluation_args_supports_single_stage_and_threshold_flags(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    test_csv = tmp_path / "dataset.csv"

    args = parse_evaluation_args(
        "Test parser",
        [
            "--run_dir",
            str(run_dir),
            "--test_csv",
            str(test_csv),
            "--single-stage",
            "--threshold_method",
            "mcc",
            "--target_recall",
            "0.9",
        ],
    )

    assert args.run_dir == run_dir
    assert args.test_csv == test_csv
    assert args.single_stage is True
    assert args.threshold_method == "mcc"
    assert args.target_recall == pytest.approx(0.9)


def test_collect_oof_predictions_reads_expected_npz_structure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = {1: np.array([0.1, 0.8]), 2: np.array([0.3, 0.9])}
    labels = {1: np.array([0, 1]), 2: np.array([0, 1])}
    np.savez(
        run_dir / "oof_predictions.npz",
        predictions=predictions,
        labels=labels,
    )

    loaded_predictions, loaded_labels = collect_oof_predictions(
        run_dir,
        logging.getLogger(__name__),
    )

    assert loaded_predictions.keys() == predictions.keys()
    assert loaded_labels.keys() == labels.keys()
    assert np.array_equal(loaded_predictions[1], predictions[1])
    assert np.array_equal(loaded_labels[2], labels[2])


def test_find_global_optimal_threshold_matches_direct_pooled_threshold() -> None:
    fold_predictions = {
        1: np.array([0.2, 0.8, 0.4]),
        2: np.array([0.7, 0.1, 0.9]),
    }
    fold_labels = {
        1: np.array([0, 1, 0]),
        2: np.array([1, 0, 1]),
    }
    target_recall = 0.8

    threshold = find_global_optimal_threshold(
        fold_predictions,
        fold_labels,
        threshold_method="recall_constrained",
        target_recall=target_recall,
        logger=logging.getLogger(__name__),
    )

    pooled_predictions = np.concatenate(list(fold_predictions.values()))
    pooled_labels = np.concatenate(list(fold_labels.values()))
    assert threshold == pytest.approx(
        find_optimal_threshold(
            pooled_predictions,
            pooled_labels,
            method="recall_constrained",
            target_recall=target_recall,
        )
    )
