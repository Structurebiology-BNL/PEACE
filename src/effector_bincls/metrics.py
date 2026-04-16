"""Package-owned metrics and threshold selection helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def high_recall_auprc(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    recall_threshold: float = 0.7,
) -> float:
    """Calculate partial AUPRC for recall values above a threshold."""
    y_true_array = np.asarray(y_true).ravel()
    y_pred_array = np.asarray(y_pred_proba).ravel()

    precision, recall, _ = precision_recall_curve(y_true_array, y_pred_array)
    high_recall_mask = recall >= recall_threshold
    if not np.any(high_recall_mask):
        logger.warning(
            "No recall values >= %s. Returning 0 for high_recall_auprc.",
            recall_threshold,
        )
        return 0.0

    high_precision = precision[high_recall_mask]
    high_recall = recall[high_recall_mask]
    sort_idx = np.argsort(high_recall)
    high_precision = high_precision[sort_idx]
    high_recall = high_recall[sort_idx]

    if len(high_recall) == 1:
        return 0.0

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz_fn(high_precision, high_recall))


def multi_scores(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Calculate binary classification metrics from probabilities."""
    y_true_array = np.asarray(y_true).ravel()
    y_pred_array = np.asarray(y_pred_proba).ravel()

    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_true_array, y_pred_array)),
        "auprc": float(average_precision_score(y_true_array, y_pred_array)),
        "high_recall_auprc_0.7": high_recall_auprc(
            y_true_array,
            y_pred_array,
            recall_threshold=0.7,
        ),
        "high_recall_auprc_0.8": high_recall_auprc(
            y_true_array,
            y_pred_array,
            recall_threshold=0.8,
        ),
    }

    if threshold is None:
        return metrics

    y_pred = (y_pred_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "accuracy": float(accuracy_score(y_true_array, y_pred)),
            "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true_array, y_pred)),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }
    )
    return metrics


def find_optimal_threshold(
    predictions: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    method: str = "youden",
    target_recall: float = 0.8,
) -> float:
    """Find an optimal classification threshold for binary probabilities."""
    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    if prediction_array.ndim > 1:
        prediction_array = prediction_array.reshape(-1)

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(label_array, prediction_array)
        optimal_idx = np.argmax(tpr - fpr)
        return float(thresholds[optimal_idx])

    if method == "f1":
        thresholds = np.linspace(0.01, 0.99, 300)
        scores = []
        for threshold in thresholds:
            y_pred = (prediction_array >= threshold).astype(int)
            scores.append(f1_score(label_array, y_pred))
        optimal_idx = np.argmax(scores)
        return float(thresholds[optimal_idx])

    if method == "mcc":
        thresholds = np.linspace(0.01, 0.99, 300)
        scores = []
        for threshold in thresholds:
            y_pred = (prediction_array >= threshold).astype(int)
            scores.append(matthews_corrcoef(label_array, y_pred))
        optimal_idx = np.argmax(scores)
        return float(thresholds[optimal_idx])

    if method == "recall_constrained":
        thresholds = np.linspace(0.001, 0.999, 1000)
        best_threshold: float | None = None
        best_precision = -1.0
        best_recall = 0.0

        for threshold in thresholds:
            y_pred = (prediction_array >= threshold).astype(int)
            recall = recall_score(label_array, y_pred)
            if recall < target_recall or y_pred.sum() == 0:
                continue

            try:
                precision = precision_score(label_array, y_pred)
            except ZeroDivisionError:
                continue

            if precision > best_precision:
                best_threshold = float(threshold)
                best_precision = float(precision)
                best_recall = float(recall)

        if best_threshold is not None:
            logger.info("Recall-constrained threshold: %.4f", best_threshold)
            logger.info(
                "Achieved recall: %.4f (target: %.4f)",
                best_recall,
                target_recall,
            )
            logger.info("Achieved precision: %.4f", best_precision)
            return best_threshold

        logger.warning(
            "Target recall %.4f not achievable. Using threshold with highest recall.",
            target_recall,
        )
        fallback_threshold: float | None = None
        best_recall = -1.0
        best_precision = 0.0

        for threshold in thresholds:
            y_pred = (prediction_array >= threshold).astype(int)
            recall = recall_score(label_array, y_pred)
            if recall <= best_recall or y_pred.sum() == 0:
                continue

            try:
                precision = precision_score(label_array, y_pred)
            except ZeroDivisionError:
                continue

            fallback_threshold = float(threshold)
            best_recall = float(recall)
            best_precision = float(precision)

        if fallback_threshold is not None:
            logger.warning("Fallback threshold: %.4f", fallback_threshold)
            logger.warning("Max achievable recall: %.4f", best_recall)
            logger.warning("Precision at fallback threshold: %.4f", best_precision)
            return fallback_threshold

        logger.error("No valid threshold found. Using default threshold 0.5.")
        return 0.5

    raise ValueError(
        "Unknown method: "
        f"{method}. Supported methods are 'youden', 'f1', 'mcc', and "
        "'recall_constrained'."
    )


def compute_multilabel_metrics(
    outputs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    thresholds: np.ndarray | list[float] | None = None,
    label_names: list[str] | None = None,
    compute_per_label: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper retained for package-internal imports if needed."""
    outputs_array = (
        outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
    )
    labels_array = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    outputs_array = np.asarray(outputs_array)
    labels_array = np.asarray(labels_array)

    if outputs_array.shape != labels_array.shape:
        raise ValueError("outputs and labels must have the same shape")

    if thresholds is None:
        thresholds_array = np.full(outputs_array.shape[1], 0.5)
    else:
        thresholds_array = np.asarray(thresholds)

    binary_predictions = outputs_array >= thresholds_array
    metrics: dict[str, Any] = {
        "micro_auroc": float(
            roc_auc_score(labels_array, outputs_array, average="micro")
        ),
        "macro_auprc": float(
            average_precision_score(labels_array, outputs_array, average="macro")
        ),
    }
    if compute_per_label:
        names = label_names or [str(idx) for idx in range(outputs_array.shape[1])]
        metrics["per_label"] = {
            name: {
                "positive_predictions": int(binary_predictions[:, idx].sum()),
                "positive_labels": int(labels_array[:, idx].sum()),
            }
            for idx, name in enumerate(names)
        }
    return metrics


__all__ = [
    "compute_multilabel_metrics",
    "find_optimal_threshold",
    "high_recall_auprc",
    "multi_scores",
]
