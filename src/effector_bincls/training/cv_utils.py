"""Package-owned utilities for cross-validation result aggregation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from effector_bincls.metrics import find_optimal_threshold, multi_scores


def save_oof_predictions(
    oof_predictions_storage: dict,
    run_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Save out-of-fold predictions to a compressed NumPy archive."""
    oof_predictions_file = run_dir / "oof_predictions.npz"
    np.savez_compressed(
        oof_predictions_file,
        predictions=dict(oof_predictions_storage["predictions"]),
        labels=dict(oof_predictions_storage["labels"]),
    )
    logger.info("Saved OOF predictions to %s", oof_predictions_file)
    return oof_predictions_file


def compute_global_threshold_optimization(
    oof_predictions_storage: dict,
    num_folds: int,
    threshold_method: str = "youden",
    target_recall: float = 0.85,
    logger: logging.Logger | None = None,
) -> dict:
    """Compute threshold optimization metrics using pooled OOF predictions."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("\n%s", "=" * 60)
    logger.info("Global Threshold Optimization")
    logger.info("%s", "=" * 60)

    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for fold in range(1, num_folds + 1):
        if fold not in oof_predictions_storage["predictions"]:
            continue
        predictions = oof_predictions_storage["predictions"][fold]
        labels = oof_predictions_storage["labels"][fold]
        if predictions is None or labels is None:
            continue
        all_predictions.append(predictions)
        all_labels.append(labels)

    if not all_predictions or not all_labels:
        logger.warning(
            "No valid OOF predictions found for global threshold optimization"
        )
        return {}

    global_predictions = np.concatenate(all_predictions, axis=0)
    global_labels = np.concatenate(all_labels, axis=0)

    logger.info(
        "Global threshold optimization using %s samples", len(global_predictions)
    )
    logger.info(
        "Positive samples: %s, Negative samples: %s",
        np.sum(global_labels),
        np.sum(1 - global_labels),
    )

    global_optimal_threshold = find_optimal_threshold(
        predictions=global_predictions,
        labels=global_labels,
        method=threshold_method,
        target_recall=target_recall,
    )
    logger.info(
        "Global optimal threshold (%s): %.4f",
        threshold_method,
        global_optimal_threshold,
    )

    global_metrics = multi_scores(
        y_true=global_labels,
        y_pred_proba=global_predictions,
        threshold=global_optimal_threshold,
    )

    logger.info("\nGlobal Threshold Metrics:")
    for metric_name, metric_value in global_metrics.items():
        if not isinstance(metric_value, (int, float)):
            continue
        if metric_name in [
            "roc_auc",
            "auprc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
        ]:
            logger.info("  %-10s: %.4f", metric_name, metric_value)
        else:
            logger.info("  %s: %s", metric_name, metric_value)

    return {
        "global_optimal_threshold": global_optimal_threshold,
        "threshold_method": threshold_method,
        "target_recall": target_recall,
        "global_metrics": global_metrics,
        "num_samples": len(global_predictions),
        "num_positive": int(np.sum(global_labels)),
        "num_negative": int(np.sum(1 - global_labels)),
    }


def compute_aggregated_metrics(
    fold_metrics: list[dict],
    metric_names: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Compute mean/std metric summaries across folds."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if metric_names is None:
        metric_names = [
            "roc_auc",
            "auprc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "mcc",
        ]

    aggregated_metrics: dict[str, float | list[float]] = {}
    for metric_name in metric_names:
        values = [
            fold_metric[metric_name]
            for fold_metric in fold_metrics
            if metric_name in fold_metric
        ]
        if not values:
            continue
        aggregated_metrics[f"{metric_name}_mean"] = float(np.mean(values))
        aggregated_metrics[f"{metric_name}_std"] = float(np.std(values))
        aggregated_metrics[f"{metric_name}_values"] = values

    logger.info("\nCross-Validation Results - Mean ± Std:")
    logger.info("\nThreshold-independent metrics:")
    for metric_name in ["roc_auc", "auprc"]:
        mean_key = f"{metric_name}_mean"
        if mean_key in aggregated_metrics:
            logger.info(
                "  %-10s: %.4f ± %.4f",
                metric_name,
                aggregated_metrics[mean_key],
                aggregated_metrics[f"{metric_name}_std"],
            )

    logger.info("\nThreshold-dependent metrics:")
    for metric_name in ["f1", "precision", "recall", "accuracy", "mcc"]:
        mean_key = f"{metric_name}_mean"
        if mean_key in aggregated_metrics:
            logger.info(
                "  %-10s: %.4f ± %.4f",
                metric_name,
                aggregated_metrics[mean_key],
                aggregated_metrics[f"{metric_name}_std"],
            )

    return aggregated_metrics


def prepare_cv_results(
    cv_results: dict,
    aggregated_metrics: dict,
    global_threshold_results: dict | None = None,
    training_mode: str = "prototype_ranking",
    num_folds: int = 5,
    config_summary: dict | None = None,
    oof_predictions_file: str | None = None,
) -> dict:
    """Prepare the serialized results payload for a CV workflow."""
    results = {
        "cv_enabled": True,
        "training_approach": training_mode,
        "cv_metrics": aggregated_metrics,
        "num_folds": num_folds,
        "fold_summary": [
            {
                "fold": fold.get("fold", index + 1),
                "val_metrics": {
                    key: value
                    for key, value in fold["val_metrics"].items()
                    if key
                    in [
                        "roc_auc",
                        "auprc",
                        "f1",
                        "precision",
                        "recall",
                        "accuracy",
                        "mcc",
                        "high_recall_auprc_0.7",
                        "high_recall_auprc_0.8",
                    ]
                },
            }
            for index, fold in enumerate(cv_results["fold_metrics"])
        ],
    }

    if global_threshold_results:
        results["global_threshold_results"] = global_threshold_results
    if config_summary:
        results["config_summary"] = config_summary
    if oof_predictions_file:
        results["oof_predictions_file"] = oof_predictions_file

    return results


__all__ = [
    "compute_aggregated_metrics",
    "compute_global_threshold_optimization",
    "prepare_cv_results",
    "save_oof_predictions",
]
