"""Package-owned cross-validation orchestration."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ml_collections import ConfigDict

from effector_bincls.models import SimplePredictor
from effector_bincls.run_utils import (
    cleanup_checkpoints,
    convert_to_serializable,
    seed_everything,
    setup_logger,
)
from effector_bincls.training.cv_utils import (
    compute_aggregated_metrics,
    compute_global_threshold_optimization,
    prepare_cv_results,
    save_oof_predictions,
)
from effector_bincls.training.data import create_two_stage_data_loader_fn
from effector_bincls.training.trainers import (
    BaselineTrainer,
    PretrainTrainer,
    PrototypeRankingTrainer,
)


def run_baseline_cv(
    config: ConfigDict,
    data_loader_fn,
    run_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Run baseline cross-validation from package-owned orchestration."""
    if run_dir is None:
        run_dir = (
            Path("results")
            / f"baseline_cv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        log_level = (
            "DEBUG" if getattr(config.hardware, "debug_logging", False) else "INFO"
        )
        logger = setup_logger(
            output_dir=run_dir, name="baseline_cv", log_level=log_level
        )

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
        train_loader, val_loader = data_loader_fn(fold)

        model = SimplePredictor(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            dropout_rate=config.model.dropout_rate,
            use_contrastive=config.model.use_contrastive,
            encoder_hidden_dim=getattr(config.model, "encoder_hidden_dim", None),
        ).to(device)
        trainer = BaselineTrainer(
            model=model,
            config=config.training,
            device=device,
            save_checkpoints=save_checkpoints,
            logger=logger,
        )
        fold_results = trainer.train_fold(
            fold_number=fold,
            train_loader=train_loader,
            val_loader=val_loader,
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
            cleanup_checkpoints(fold_dir, logger, fold, checkpoint_paths)

    oof_predictions_file = save_oof_predictions(
        oof_predictions_storage, run_dir, logger
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
        fold_metrics=fold_metrics, logger=logger
    )
    aggregated_metrics["threshold_mean"] = np.mean(fold_thresholds)
    aggregated_metrics["threshold_std"] = np.std(fold_thresholds)
    aggregated_metrics["threshold_values"] = fold_thresholds

    results = prepare_cv_results(
        cv_results=cv_results,
        aggregated_metrics=aggregated_metrics,
        global_threshold_results=global_threshold_results,
        training_mode="Single-stage Baseline",
        num_folds=num_folds,
        config_summary={
            "model_type": config.model.type,
            "input_dim": config.model.input_dim,
            "num_folds": num_folds,
            "random_seed": config.hardware.random_seed,
            "training_mode": "Single-stage Baseline",
            "loss_type": "bce",
        },
        oof_predictions_file=str(oof_predictions_file),
    )
    results["baseline_cv_enabled"] = True
    results["threshold_method"] = threshold_method
    results["target_recall"] = target_recall
    for index, fold in enumerate(cv_results["fold_metrics"]):
        results["fold_summary"][index].update(
            {
                "optimal_threshold": fold["optimal_threshold"],
                "threshold_method": fold.get("threshold_method", threshold_method),
                "epochs_trained": fold["epochs_trained"],
            }
        )
    return convert_to_serializable(results)


def run_prototype_ranking_cv(
    config: ConfigDict,
    data_loader_fn,
    run_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Run single-stage prototype ranking cross-validation."""
    if run_dir is None:
        run_dir = (
            Path("results")
            / f"prototype_ranking_cv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        log_level = (
            "DEBUG" if getattr(config.hardware, "debug_logging", False) else "INFO"
        )
        logger = setup_logger(
            output_dir=run_dir, name="prototype_ranking_cv", log_level=log_level
        )

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
        train_loader, val_loader = data_loader_fn(fold)

        model = SimplePredictor(
            input_dim=config.model.input_dim,
            output_dim=1,
            dropout_rate=getattr(config.model, "dropout_rate", 0.2),
            use_contrastive=True,
            contrastive_dim=getattr(config.model, "contrastive_dim", 128),
            encoder_hidden_dim=getattr(
                config.model, "encoder_hidden_dim", config.model.input_dim
            ),
        ).to(device)
        trainer = PrototypeRankingTrainer(
            model=model,
            config=config.training,
            device=device,
            save_checkpoints=save_checkpoints,
            logger=logger,
        )
        fold_results = trainer.train_fold(
            fold_number=fold,
            train_loader=train_loader,
            val_loader=val_loader,
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
            cleanup_checkpoints(fold_dir, logger, fold, checkpoint_paths)

    oof_predictions_file = save_oof_predictions(
        oof_predictions_storage, run_dir, logger
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
        fold_metrics=fold_metrics, logger=logger
    )
    aggregated_metrics["threshold_mean"] = np.mean(fold_thresholds)
    aggregated_metrics["threshold_std"] = np.std(fold_thresholds)
    aggregated_metrics["threshold_values"] = fold_thresholds

    results = prepare_cv_results(
        cv_results=cv_results,
        aggregated_metrics=aggregated_metrics,
        global_threshold_results=global_threshold_results,
        training_mode="Single-stage Prototype Ranking",
        num_folds=num_folds,
        config_summary={
            "model_type": config.model.type,
            "input_dim": config.model.input_dim,
            "num_folds": num_folds,
            "random_seed": config.hardware.random_seed,
            "training_mode": "Single-stage Prototype Ranking",
            "contrastive_type": "prototype_ranking",
        },
        oof_predictions_file=str(oof_predictions_file),
    )
    results["prototype_ranking_cv_enabled"] = True
    results["threshold_method"] = threshold_method
    results["target_recall"] = target_recall
    for index, fold in enumerate(cv_results["fold_metrics"]):
        results["fold_summary"][index].update(
            {
                "optimal_threshold": fold["optimal_threshold"],
                "threshold_method": fold.get("threshold_method", threshold_method),
                "epochs_trained": fold["epochs_trained"],
            }
        )
    return convert_to_serializable(results)


def run_prototype_ranking_two_stage_cv(
    config: ConfigDict,
    run_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Run two-stage prototype ranking cross-validation."""
    if run_dir is None:
        run_dir = Path("results") / (
            f"prototype_ranking_two_stage_cv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        log_level = (
            "DEBUG" if getattr(config.hardware, "debug_logging", False) else "INFO"
        )
        logger = setup_logger(
            output_dir=run_dir,
            name="prototype_ranking_two_stage_cv",
            log_level=log_level,
        )

    seed_everything(config.hardware.random_seed, config.hardware.deterministic)
    device = (
        f"cuda:{config.hardware.gpu_id}"
        if config.hardware.gpu_id >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    num_folds = config.training.num_folds
    save_checkpoints = config.output.save_checkpoints
    pretrained_run_dir = getattr(config.training, "run_dir", None)
    skip_pretraining = pretrained_run_dir is not None
    if pretrained_run_dir is not None:
        pretrained_run_dir = Path(pretrained_run_dir)
        if not pretrained_run_dir.exists():
            raise FileNotFoundError(
                f"Pretrained model directory not found: {pretrained_run_dir}"
            )

    data_loader_fn = create_two_stage_data_loader_fn(config, logger)
    cv_results = {"fold_metrics": [], "config": config}
    oof_predictions_storage = {"predictions": {}, "labels": {}}

    for fold in range(1, num_folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model = SimplePredictor(
            input_dim=config.model.input_dim,
            output_dim=1,
            dropout_rate=getattr(config.model, "dropout_rate", 0.2),
            use_contrastive=True,
            contrastive_dim=getattr(config.model, "contrastive_dim", 128),
            encoder_hidden_dim=getattr(
                config.model,
                "encoder_hidden_dim",
                config.model.input_dim,
            ),
        ).to(device)

        if not skip_pretraining:
            pretraining_dir = fold_dir / "pretraining"
            pretrain_trainer = PretrainTrainer(
                model=model,
                config=config.training.pretraining,
                device=device,
                save_checkpoints=True,
                logger=logger,
            )
            pretrain_train_loader, pretrain_val_loader = data_loader_fn(
                fold,
                "pretraining",
            )
            pretraining_results = pretrain_trainer.train(
                train_loader=pretrain_train_loader,
                val_loader=pretrain_val_loader,
                save_dir=pretraining_dir,
                plot_curves=config.output.plot_training_curves,
                stage=f"fold_{fold}_pretraining",
            )
            pretrained_checkpoint_path = pretraining_results.get("best_checkpoint_path")
            if (
                not pretrained_checkpoint_path
                or not Path(pretrained_checkpoint_path).exists()
            ):
                raise RuntimeError(
                    f"Pretraining failed to save checkpoint for fold {fold}"
                )
        else:
            pretraining_results = {}
            pretrained_checkpoint_path = None

        finetuning_dir = fold_dir / "finetuning"
        if not skip_pretraining:
            checkpoint = torch.load(
                pretrained_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"])
            finetune_trainer = PrototypeRankingTrainer(
                model=model,
                config=config.training.finetuning,
                device=device,
                save_checkpoints=save_checkpoints,
                logger=logger,
            )
            if checkpoint.get("prototypes") is not None:
                finetune_trainer.prototypes = checkpoint["prototypes"].to(device)
                finetune_trainer._sync_prototypes_to_loss_fn()
        else:
            finetune_trainer = PrototypeRankingTrainer(
                model=model,
                config=config.training.finetuning,
                device=device,
                save_checkpoints=save_checkpoints,
                logger=logger,
            )

        finetune_train_loader, finetune_val_loader = data_loader_fn(
            fold,
            "finetuning",
        )
        finetuning_results = finetune_trainer.train_fold(
            fold_number=fold,
            train_loader=finetune_train_loader,
            val_loader=finetune_val_loader,
            save_dir=finetuning_dir,
            plot_curves=config.output.plot_training_curves,
            threshold_method=config.training.threshold_method,
            target_recall=config.training.target_recall,
            pretrained_run_dir=pretrained_run_dir if skip_pretraining else None,
        )

        fold_results = {
            "fold": fold,
            "pretraining_epochs": pretraining_results.get("epochs_trained", 0),
            "finetuning_epochs": finetuning_results.get("epochs_trained", 0),
            "pretraining_checkpoint": (
                str(pretrained_checkpoint_path) if pretrained_checkpoint_path else None
            ),
            "val_metrics": finetuning_results.get("val_metrics", {}),
            "val_predictions": finetuning_results.get("val_predictions"),
            "val_labels": finetuning_results.get("val_labels"),
            "optimal_threshold": finetuning_results.get("optimal_threshold"),
            "threshold_method": finetuning_results.get("threshold_method"),
        }
        cv_results["fold_metrics"].append(fold_results)
        oof_predictions_storage["predictions"][fold] = finetuning_results.get(
            "val_predictions"
        )
        oof_predictions_storage["labels"][fold] = finetuning_results.get("val_labels")

        if not save_checkpoints:
            checkpoint_paths: list[str | Path] = []
            if not skip_pretraining and pretrained_checkpoint_path:
                checkpoint_paths.append(pretrained_checkpoint_path)
            if "best_checkpoint_path" in finetuning_results:
                checkpoint_paths.append(finetuning_results["best_checkpoint_path"])
            cleanup_checkpoints(fold_dir, logger, fold, checkpoint_paths)

    oof_predictions_file = save_oof_predictions(
        oof_predictions_storage, run_dir, logger
    )
    threshold_method = getattr(config.training, "threshold_method", "youden")
    target_recall = getattr(config.training, "target_recall", 0.85)
    global_threshold_results = compute_global_threshold_optimization(
        oof_predictions_storage=oof_predictions_storage,
        num_folds=num_folds,
        threshold_method=threshold_method,
        target_recall=target_recall,
        logger=logger,
    )
    aggregated_metrics = compute_aggregated_metrics(
        fold_metrics=[
            fold_result["val_metrics"] for fold_result in cv_results["fold_metrics"]
        ],
        logger=logger,
    )
    fold_thresholds = [
        fold_result["optimal_threshold"]
        for fold_result in cv_results["fold_metrics"]
        if fold_result["optimal_threshold"] is not None
    ]
    if fold_thresholds:
        aggregated_metrics["threshold_mean"] = float(np.mean(fold_thresholds))
        aggregated_metrics["threshold_std"] = float(np.std(fold_thresholds))
        aggregated_metrics["threshold_values"] = fold_thresholds

    config_summary = {
        "model_type": config.model.type,
        "input_dim": config.model.input_dim,
        "num_folds": num_folds,
        "random_seed": config.hardware.random_seed,
        "pretraining_csv": config.data.pretraining_csv_path,
        "finetuning_csv": config.data.finetuning_csv_path,
    }
    if skip_pretraining:
        config_summary["pretrained_run_dir"] = str(pretrained_run_dir)
        config_summary["training_mode"] = "finetuning_only"

    results = prepare_cv_results(
        cv_results=cv_results,
        aggregated_metrics=aggregated_metrics,
        global_threshold_results=global_threshold_results,
        training_mode="two_stage_prototype_ranking",
        num_folds=num_folds,
        config_summary=config_summary,
        oof_predictions_file=str(oof_predictions_file),
    )
    results["two_stage_cv_enabled"] = True
    results["finetuning_only_mode"] = skip_pretraining
    results["threshold_method"] = threshold_method
    results["target_recall"] = target_recall
    for index, fold in enumerate(cv_results["fold_metrics"]):
        results["fold_summary"][index].update(
            {
                "pretraining_epochs": fold["pretraining_epochs"],
                "finetuning_epochs": fold["finetuning_epochs"],
                "optimal_threshold": fold["optimal_threshold"],
                "threshold_method": fold.get("threshold_method", threshold_method),
            }
        )
    return convert_to_serializable(results)
