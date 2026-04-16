"""Shared helpers for package-native evaluation entrypoints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from effector_bincls.metrics import find_optimal_threshold


def parse_evaluation_args(
    description: str = "Test evaluation script",
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command line arguments for labeled-data evaluation."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        required=True,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        default=False,
        help=(
            "Use single-stage training checkpoint paths (fold_X/checkpoint.pt). "
            "Default is two-stage (fold_X/finetuning/checkpoint.pt)."
        ),
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="youden",
        choices=["youden", "f1", "mcc", "recall_constrained"],
        help="Method for global threshold optimization",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.8,
        help="Target recall for recall_constrained threshold method",
    )
    return parser.parse_args(argv)


def collect_oof_predictions(
    run_dir: Path,
    logger: logging.Logger,
    prediction_key: str = "predictions",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load saved out-of-fold predictions and labels from a run directory."""
    oof_predictions_file = run_dir / "oof_predictions.npz"
    if not oof_predictions_file.exists():
        raise FileNotFoundError(
            f"OOF predictions file not found: {oof_predictions_file}"
        )

    logger.info("Loading OOF predictions from %s", oof_predictions_file)
    data = np.load(oof_predictions_file, allow_pickle=True)
    fold_predictions = data[prediction_key].item()
    fold_labels = data["labels"].item()

    total_samples = sum(len(preds) for preds in fold_predictions.values())
    logger.info("Loaded OOF predictions from %s folds", len(fold_predictions))
    logger.info("Total OOF samples: %s", total_samples)
    return fold_predictions, fold_labels


def pool_fold_predictions(
    fold_predictions: dict[int, np.ndarray],
    fold_labels: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate fold prediction dictionaries into pooled 1D arrays."""
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for fold in sorted(fold_predictions):
        preds = fold_predictions[fold]
        labels = fold_labels[fold]
        if preds.ndim > 1:
            preds = preds.ravel()
        if labels.ndim > 1:
            labels = labels.ravel()
        all_predictions.append(preds.astype(np.float64))
        all_labels.append(labels.astype(np.int64))

    return np.concatenate(all_predictions), np.concatenate(all_labels)


def find_global_optimal_threshold(
    fold_predictions: dict[int, np.ndarray],
    fold_labels: dict[int, np.ndarray],
    threshold_method: str,
    target_recall: float,
    logger: logging.Logger,
) -> float:
    """Find a global threshold from pooled OOF predictions."""
    logger.info(
        "Finding global optimal threshold using %s method...",
        threshold_method,
    )
    pooled_predictions, pooled_labels = pool_fold_predictions(
        fold_predictions,
        fold_labels,
    )
    logger.info("Combined %s predictions across all folds", len(pooled_predictions))
    logger.info(
        "Predictions shape: %s, dtype: %s",
        pooled_predictions.shape,
        pooled_predictions.dtype,
    )
    logger.info(
        "Labels shape: %s, dtype: %s",
        pooled_labels.shape,
        pooled_labels.dtype,
    )
    logger.info("Unique labels: %s", np.unique(pooled_labels))
    logger.info(
        "Prediction range: [%.4f, %.4f]",
        pooled_predictions.min(),
        pooled_predictions.max(),
    )

    optimal_threshold = find_optimal_threshold(
        pooled_predictions,
        pooled_labels,
        method=threshold_method,
        target_recall=target_recall,
    )
    logger.info("Global optimal threshold: %.4f", optimal_threshold)
    return float(optimal_threshold)


def add_high_recall_summary(results: dict[str, Any]) -> None:
    """Attach grouped high-recall AUPRC summary when the metrics exist."""
    test_metrics = results.get("test_metrics", {})
    if "high_recall_auprc_0.7" not in test_metrics:
        return
    results["high_recall_auprc"] = {
        "recall_0.7": test_metrics["high_recall_auprc_0.7"],
        "recall_0.8": test_metrics["high_recall_auprc_0.8"],
    }


__all__ = [
    "add_high_recall_summary",
    "collect_oof_predictions",
    "find_global_optimal_threshold",
    "parse_evaluation_args",
    "pool_fold_predictions",
]
