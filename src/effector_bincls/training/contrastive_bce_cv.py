"""Cross-validation orchestration for the public Contrastive-BCE workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from effector_bincls.models import SimplePredictor
from effector_bincls.run_utils import (
    cleanup_checkpoints,
    convert_to_serializable,
)
from effector_bincls.training.contrastive_bce_trainer import (
    ContrastiveBCETrainer,
)
from effector_bincls.training.cv_utils import (
    compute_aggregated_metrics,
    compute_global_threshold_optimization,
    prepare_cv_results,
    save_oof_predictions,
)


def run_contrastive_bce_cv(
    config: ConfigDict,
    data_loader_fn,
    run_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Run Contrastive-BCE cross-validation and pooled OOF thresholding."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{config.hardware.gpu_id}" if config.hardware.gpu_id >= 0 else "cpu"
    num_folds = config.training.num_folds
    save_checkpoints = getattr(config.output, "save_checkpoints", False)
    threshold_method = getattr(config.training, "threshold_method", "youden")
    target_recall = getattr(config.training, "target_recall", 0.85)

    cv_results = {"fold_metrics": [], "config": config}
    oof_predictions_storage = {"predictions": {}, "labels": {}}

    for fold in range(1, num_folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_loader, validation_loader = data_loader_fn(fold)

        model = SimplePredictor(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            dropout_rate=config.model.dropout_rate,
            use_contrastive=True,
            contrastive_dim=config.model.contrastive_dim,
            encoder_hidden_dim=config.model.encoder_hidden_dim,
        ).to(device)
        trainer = ContrastiveBCETrainer(
            model=model,
            config=config.training,
            device=device,
            save_checkpoints=save_checkpoints,
            logger=logger,
        )
        fold_results = trainer.train_fold(
            fold_number=fold,
            train_loader=train_loader,
            val_loader=validation_loader,
            save_dir=fold_dir,
            plot_curves=getattr(config.output, "plot_training_curves", True),
            threshold_method=threshold_method,
            target_recall=target_recall,
        )
        cv_results["fold_metrics"].append(fold_results)
        oof_predictions_storage["predictions"][fold] = fold_results["val_predictions"]
        oof_predictions_storage["labels"][fold] = fold_results["val_labels"]

        if not save_checkpoints:
            checkpoint_paths = []
            if "best_checkpoint_path" in fold_results:
                checkpoint_paths.append(fold_results["best_checkpoint_path"])
            cleanup_checkpoints(
                fold_dir,
                logger,
                fold,
                checkpoint_paths,
            )

    oof_predictions_file = save_oof_predictions(
        oof_predictions_storage,
        run_dir,
        logger,
    )
    global_threshold_results = compute_global_threshold_optimization(
        oof_predictions_storage=oof_predictions_storage,
        num_folds=num_folds,
        threshold_method=threshold_method,
        target_recall=target_recall,
        logger=logger,
    )
    fold_metrics = [
        fold_result["val_metrics"] for fold_result in cv_results["fold_metrics"]
    ]
    fold_thresholds = [
        fold_result["optimal_threshold"] for fold_result in cv_results["fold_metrics"]
    ]
    aggregated_metrics = compute_aggregated_metrics(
        fold_metrics=fold_metrics,
        logger=logger,
    )
    aggregated_metrics["threshold_mean"] = np.mean(fold_thresholds)
    aggregated_metrics["threshold_std"] = np.std(fold_thresholds)
    aggregated_metrics["threshold_values"] = fold_thresholds

    results = prepare_cv_results(
        cv_results=cv_results,
        aggregated_metrics=aggregated_metrics,
        global_threshold_results=global_threshold_results,
        training_mode="Single-stage Contrastive BCE",
        num_folds=num_folds,
        config_summary={
            "model_type": config.model.type,
            "input_dim": config.model.input_dim,
            "num_folds": num_folds,
            "random_seed": config.hardware.random_seed,
            "training_mode": "Single-stage Contrastive BCE",
            "loss_type": "contrastive_bce",
        },
        oof_predictions_file=str(oof_predictions_file),
    )
    results["contrastive_bce_cv_enabled"] = True
    results["threshold_method"] = threshold_method
    results["target_recall"] = target_recall
    for index, fold_result in enumerate(cv_results["fold_metrics"]):
        results["fold_summary"][index].update(
            {
                "optimal_threshold": fold_result["optimal_threshold"],
                "threshold_method": fold_result.get(
                    "threshold_method",
                    threshold_method,
                ),
                "epochs_trained": fold_result["epochs_trained"],
            }
        )
    return convert_to_serializable(results)


__all__ = ["run_contrastive_bce_cv"]
