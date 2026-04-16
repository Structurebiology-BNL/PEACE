"""Package-owned plotting helpers shared across workflows and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from effector_bincls.evaluation.reporting import (
    plot_test_metrics,
    plot_threshold_analysis,
)


def plot_training_curves(
    metrics_tracker: Any,
    fold: int | None = None,
    save_dir: str | Path | None = None,
    stage: str | None = None,
) -> None:
    """Plot and save training curves for the current fold or stage."""
    if save_dir is None:
        return

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_str = str(fold) if fold is not None else "global"
    fold_display = f"Fold {fold}" if fold is not None else "Global Training"
    stage_suffix = f"_stage_{stage}" if stage else ""
    title_suffix = f" - {stage.capitalize()} Stage" if stage else ""

    train_losses = [metric.get("loss", 0) for metric in metrics_tracker.train_metrics]
    val_losses = [metric.get("loss", 0) for metric in metrics_tracker.val_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training and Validation Loss - {fold_display}{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_dir / f"loss_fold_{fold_str}{stage_suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    if not metrics_tracker.train_metrics or not metrics_tracker.val_metrics:
        return

    available_metrics: list[tuple[str, str]] = []
    first_train = metrics_tracker.train_metrics[0]

    if "roc_auc" in first_train:
        available_metrics.append(("roc_auc", "ROC AUC"))
    if "auprc" in first_train:
        available_metrics.append(("auprc", "AUPRC"))

    for prefix in ["micro_", "macro_"]:
        for metric in ["auroc", "auprc"]:
            metric_name = f"{prefix}{metric}"
            if metric_name in first_train:
                available_metrics.append(
                    (metric_name, f"{prefix.upper()}{metric.upper()}")
                )

    for metric_name, display_name in available_metrics:
        plt.figure(figsize=(10, 6))
        train_metric = [
            metric.get(metric_name, 0) for metric in metrics_tracker.train_metrics
        ]
        val_metric = [
            metric.get(metric_name, 0) for metric in metrics_tracker.val_metrics
        ]

        plt.plot(train_metric, label=f"Train {display_name}")
        plt.plot(val_metric, label=f"Validation {display_name}")
        plt.title(f"{display_name} - {fold_display}{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel(display_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_dir / f"{metric_name}_fold_{fold_str}{stage_suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


__all__ = [
    "plot_test_metrics",
    "plot_threshold_analysis",
    "plot_training_curves",
]
