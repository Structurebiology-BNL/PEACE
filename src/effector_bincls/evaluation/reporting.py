"""Plotting helpers for package-native evaluation entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from effector_bincls.metrics import find_optimal_threshold, high_recall_auprc


def _find_roc_marker_index(
    roc_thresholds: np.ndarray,
    threshold: float,
) -> int | None:
    """Return the nearest finite ROC threshold index for marker plotting."""
    if not np.isfinite(threshold):
        return None

    finite_indices = np.flatnonzero(np.isfinite(roc_thresholds))
    if len(finite_indices) == 0:
        return None

    finite_thresholds = roc_thresholds[finite_indices]
    nearest_position = int(np.argmin(np.abs(finite_thresholds - threshold)))
    return int(finite_indices[nearest_position])


def plot_test_metrics(test_metrics: dict[str, Any], save_dir: str | Path) -> None:
    """Plot a small test-metrics bar chart for saved evaluation results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(test_metrics, dict) and "test_metrics" in test_metrics:
        metrics_dict = test_metrics["test_metrics"]
    else:
        metrics_dict = test_metrics

    available_metrics = []
    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "auprc",
        "mcc",
    ]:
        if metric in metrics_dict:
            available_metrics.append(metric)

    if not available_metrics:
        for prefix in ["micro_", "macro_"]:
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
                "auprc",
                "mcc",
            ]:
                metric_name = f"{prefix}{metric}"
                if metric_name in metrics_dict:
                    available_metrics.append(metric_name)

    if not available_metrics:
        return

    plt.figure(figsize=(12, 6))
    x = np.arange(len(available_metrics))
    bars = plt.bar(x, [metrics_dict[metric] for metric in available_metrics], width=0.6)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Test Set Performance Metrics")
    plt.xticks(x, [metric.upper() for metric in available_metrics])

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{metrics_dict[available_metrics[idx]]:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_dir / "test_metrics.png")
    plt.close()


def plot_threshold_analysis(
    outputs: np.ndarray | torch.Tensor | tuple[Any, ...],
    labels: np.ndarray | torch.Tensor,
    save_dir: str | Path,
    fold_number: str | int | None = None,
    threshold_methods: list[str] | None = None,
    target_recalls: list[float] | None = None,
    optimal_threshold: float | None = None,
    threshold_method_used: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Plot ROC and precision-recall curves with threshold markers."""
    try:
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy().ravel()
        else:
            labels_np = np.asarray(labels).ravel()

        if logits.ndim == 3:
            logits = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        probs = np.asarray(logits).ravel()
        if np.max(probs) > 1.0 or np.min(probs) < 0.0:
            probs = 1.0 / (1.0 + np.exp(-probs))

        if threshold_methods is None:
            threshold_methods = ["f1", "mcc", "youden", "recall_constrained"]
        if target_recalls is None:
            target_recalls = [0.7, 0.8]

        threshold_methods_with_default = ["default_0.5", *threshold_methods]
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fold_str = f"_fold_{fold_number}" if fold_number is not None else ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        fpr, tpr, roc_thresholds = roc_curve(labels_np, probs)
        roc_auc = roc_auc_score(labels_np, probs)
        ax1.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})",
        )
        ax1.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )

        precision, recall, pr_thresholds = precision_recall_curve(labels_np, probs)
        avg_precision = average_precision_score(labels_np, probs)
        hr_auprc_07 = high_recall_auprc(labels_np, probs, recall_threshold=0.7)
        hr_auprc_08 = high_recall_auprc(labels_np, probs, recall_threshold=0.8)
        label = f"PR curve (AUPRC = {avg_precision:.3f})"
        label += (
            f"\nHigh-recall AUPRC: R>=0.7={hr_auprc_07:.3f}, R>=0.8={hr_auprc_08:.3f}"
        )
        ax2.plot(recall, precision, color="darkorange", lw=2, label=label)

        baseline_precision = np.sum(labels_np) / len(labels_np)
        ax2.axhline(
            y=baseline_precision,
            color="navy",
            linestyle="--",
            lw=2,
            label=f"Random classifier (P = {baseline_precision:.3f})",
        )

        colors = ["red", "green", "blue", "purple", "orange"]
        threshold_info: list[dict[str, float | str]] = []

        if optimal_threshold is not None:
            roc_idx = _find_roc_marker_index(roc_thresholds, optimal_threshold)
            y_pred = (probs >= optimal_threshold).astype(int)
            achieved_recall = recall_score(labels_np, y_pred, zero_division=0)
            achieved_precision = precision_score(labels_np, y_pred, zero_division=0)
            if roc_idx is not None and roc_idx < len(fpr):
                ax1.plot(
                    fpr[roc_idx],
                    tpr[roc_idx],
                    "*",
                    color="black",
                    markersize=15,
                    markeredgewidth=2,
                    markeredgecolor="white",
                    label=(
                        f"USED {threshold_method_used or 'threshold'}: "
                        f"T={optimal_threshold:.3f}"
                    ),
                )
            ax2.plot(
                achieved_recall,
                achieved_precision,
                "*",
                color="black",
                markersize=15,
                markeredgewidth=2,
                markeredgecolor="white",
                label=(
                    f"USED {threshold_method_used or 'threshold'}: "
                    f"T={optimal_threshold:.3f}"
                ),
            )
            threshold_info.append(
                {
                    "method": f"USED_{threshold_method_used or 'threshold'}",
                    "threshold": optimal_threshold,
                    "precision": float(achieved_precision),
                    "recall": float(achieved_recall),
                }
            )
            if logger is not None:
                logger.info(
                    (
                        "Marked optimal threshold on plot: %.4f "
                        "(method: %s, P=%.3f, R=%.3f)"
                    ),
                    optimal_threshold,
                    threshold_method_used,
                    achieved_precision,
                    achieved_recall,
                )

        for index, method in enumerate(threshold_methods_with_default):
            color = (
                "red" if method == "default_0.5" else colors[(index - 1) % len(colors)]
            )
            if method == "default_0.5":
                default_threshold = 0.5
                y_pred = (probs >= default_threshold).astype(int)
                achieved_recall = recall_score(labels_np, y_pred, zero_division=0)
                achieved_precision = precision_score(labels_np, y_pred, zero_division=0)
                roc_idx = _find_roc_marker_index(roc_thresholds, default_threshold)
                if roc_idx is not None and roc_idx < len(fpr):
                    ax1.plot(
                        fpr[roc_idx],
                        tpr[roc_idx],
                        "o",
                        color=color,
                        markersize=10,
                        alpha=0.8,
                        label=f"{method}: T=0.5 (EffectorP default)",
                    )
                ax2.plot(
                    achieved_recall,
                    achieved_precision,
                    "o",
                    color=color,
                    markersize=10,
                    alpha=0.8,
                    label=f"{method}: T=0.5 (EffectorP default)",
                )
                threshold_info.append(
                    {
                        "method": method,
                        "threshold": default_threshold,
                        "precision": float(achieved_precision),
                        "recall": float(achieved_recall),
                    }
                )
                continue

            if method == "recall_constrained":
                for target_recall in target_recalls:
                    try:
                        optimal_thresh = find_optimal_threshold(
                            probs,
                            labels_np,
                            method=method,
                            target_recall=target_recall,
                        )
                    except Exception as exc:
                        if logger is not None:
                            logger.warning(
                                (
                                    "Error calculating %s threshold "
                                    "for target_recall=%s: %s"
                                ),
                                method,
                                target_recall,
                                exc,
                            )
                        continue

                    roc_idx = _find_roc_marker_index(roc_thresholds, optimal_thresh)
                    y_pred = (probs >= optimal_thresh).astype(int)
                    achieved_recall = recall_score(labels_np, y_pred, zero_division=0)
                    achieved_precision = precision_score(
                        labels_np, y_pred, zero_division=0
                    )
                    if roc_idx is not None and roc_idx < len(fpr):
                        ax1.plot(
                            fpr[roc_idx],
                            tpr[roc_idx],
                            "o",
                            color=color,
                            markersize=8,
                            alpha=0.7,
                            label=(
                                f"{method} (R={target_recall}): T={optimal_thresh:.3f}"
                            ),
                        )
                    ax2.plot(
                        achieved_recall,
                        achieved_precision,
                        "o",
                        color=color,
                        markersize=8,
                        alpha=0.7,
                        label=f"{method} (R={target_recall}): T={optimal_thresh:.3f}",
                    )
                    threshold_info.append(
                        {
                            "method": f"{method}_R{target_recall}",
                            "threshold": float(optimal_thresh),
                            "precision": float(achieved_precision),
                            "recall": float(achieved_recall),
                        }
                    )
                continue

            try:
                optimal_thresh = find_optimal_threshold(probs, labels_np, method=method)
            except Exception as exc:
                if logger is not None:
                    logger.warning("Error calculating %s threshold: %s", method, exc)
                continue

            roc_idx = _find_roc_marker_index(roc_thresholds, optimal_thresh)
            y_pred = (probs >= optimal_thresh).astype(int)
            achieved_recall = recall_score(labels_np, y_pred, zero_division=0)
            achieved_precision = precision_score(labels_np, y_pred, zero_division=0)
            if roc_idx is not None and roc_idx < len(fpr):
                ax1.plot(
                    fpr[roc_idx],
                    tpr[roc_idx],
                    "o",
                    color=color,
                    markersize=8,
                    label=f"{method}: T={optimal_thresh:.3f}",
                )
            ax2.plot(
                achieved_recall,
                achieved_precision,
                "o",
                color=color,
                markersize=8,
                label=f"{method}: T={optimal_thresh:.3f}",
            )
            threshold_info.append(
                {
                    "method": method,
                    "threshold": float(optimal_thresh),
                    "precision": float(achieved_precision),
                    "recall": float(achieved_recall),
                }
            )

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve with Optimal Thresholds")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve with Optimal Thresholds")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filename = f"threshold_analysis{fold_str}.png"
        plt.savefig(save_dir / plot_filename, dpi=150, bbox_inches="tight")
        plt.close()

        if threshold_info:
            import pandas as pd

            csv_filename = f"threshold_analysis{fold_str}.csv"
            pd.DataFrame(threshold_info).to_csv(save_dir / csv_filename, index=False)
            if logger is not None:
                logger.info("Threshold analysis saved to %s", save_dir / csv_filename)

        if logger is not None:
            logger.info(
                "Threshold analysis plots saved to %s",
                save_dir / plot_filename,
            )
    except Exception as exc:
        if logger is not None:
            logger.warning("Error in threshold analysis plotting: %s", exc)


__all__ = ["plot_test_metrics", "plot_threshold_analysis"]
