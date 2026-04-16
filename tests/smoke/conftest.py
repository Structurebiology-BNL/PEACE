from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from effector_bincls.data import write_packed_embedding_dataset
from effector_bincls.models import SimplePredictor

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def copy_binary_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "binary_dataset.csv"
    shutil.copy(FIXTURE_DIR / "binary_dataset.csv", dataset_path)
    return dataset_path


def create_embedding_dir(tmp_path: Path, input_dim: int = 8) -> Path:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    sequence_ids = [f"seq{index}" for index in range(6)]
    embeddings = np.asarray(
        [
            np.stack(
                [
                    np.full(input_dim, fill_value=float(index + 101), dtype=np.float32),
                    np.full(input_dim, fill_value=float(index + 1), dtype=np.float32),
                ],
                axis=0,
            )
            for index in range(6)
        ],
        dtype=np.float32,
    )
    write_packed_embedding_dataset(
        embedding_dir,
        sequence_ids,
        embeddings,
        pooling_type="mean",
        original_variant_index=1,
    )

    return embedding_dir


def write_config(config: dict[str, Any], path: Path) -> Path:
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def latest_run_dir(results_dir: Path, model_type: str) -> Path:
    run_dirs = sorted((results_dir / model_type).glob("run_*"))
    if not run_dirs:
        raise AssertionError(
            f"No run directories found under {results_dir / model_type}"
        )
    return run_dirs[-1]


def create_historical_run_dir(
    tmp_path: Path,
    *,
    is_single_stage: bool,
    config_overrides: dict[str, Any] | None = None,
    prototype_in_extra_state: bool = False,
) -> tuple[Path, Path]:
    input_dim = 8
    contrastive_dim = 4
    model = SimplePredictor(
        input_dim=input_dim,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=contrastive_dim,
        encoder_hidden_dim=input_dim,
    )
    model.set_training_mode("pretraining")

    run_dir = tmp_path / ("single_stage_run" if is_single_stage else "two_stage_run")
    run_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = create_embedding_dir(tmp_path, input_dim=input_dim)

    config: dict[str, Any] = {
        "data": {
            "embedding_dir": str(embedding_dir),
            "results_dir": str(tmp_path / "results"),
        },
        "features": {
            "normalize": True,
            "pooling_type": "mean",
        },
        "model": {
            "type": "simple",
            "input_dim": input_dim,
            "output_dim": 1,
            "dropout_rate": 0.1,
            "contrastive_dim": contrastive_dim,
            "encoder_hidden_dim": input_dim,
        },
        "training": {
            "num_folds": 2,
            "use_variants": False,
            "prototype_loss_params": {
                "scoring_temperature": 0.7,
            },
            "finetuning": {
                "prototype_loss_params": {
                    "scoring_temperature": 0.35,
                }
            },
        },
        "hardware": {
            "gpu_id": -1,
            "random_seed": 42,
            "deterministic": True,
            "debug_logging": False,
        },
    }
    if config_overrides:
        merge_nested_dict(config, config_overrides)

    write_config(config, run_dir / "config.yml")

    for fold in (1, 2):
        fold_dir = run_dir / f"fold_{fold}"
        checkpoint_dir = fold_dir if is_single_stage else fold_dir / "finetuning"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state": model.state_dict(),
        }
        prototypes = torch.randn(2, contrastive_dim)
        if prototype_in_extra_state:
            checkpoint["extra_state"] = {"prototypes": prototypes}
        else:
            checkpoint["prototypes"] = prototypes

        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

    return run_dir, embedding_dir


def make_baseline_config(
    tmp_path: Path,
    dataset_path: Path,
    embedding_dir: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "csv_path": str(dataset_path),
            "embedding_dir": str(embedding_dir),
            "results_dir": str(tmp_path / "baseline_results"),
        },
        "features": {
            "normalize": True,
            "pooling_type": "mean",
        },
        "model": {
            "type": "simple_predictor",
            "input_dim": 8,
            "output_dim": 1,
            "dropout_rate": 0.1,
            "use_contrastive": False,
            "encoder_hidden_dim": 8,
        },
        "training": {
            "batch_size": 2,
            "num_folds": 2,
            "threshold_method": "youden",
            "target_recall": 0.8,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "early_stopping_patience": 1,
            "grad_clip_value": 1.0,
            "use_variants": False,
            "loss_type": "bce",
            "label_smoothing": 0.0,
            "monitor_metric": "roc_auc",
            "mode": "max",
            "lr_scheduler": {
                "scheduler_type": "plateau",
                "patience": 1,
            },
        },
        "output": {
            "save_checkpoints": True,
            "plot_training_curves": False,
        },
        "hardware": {
            "gpu_id": -1,
            "random_seed": 42,
            "deterministic": True,
            "debug_logging": False,
            "num_workers": 0,
        },
    }


def make_prototype_single_config(
    tmp_path: Path,
    dataset_path: Path,
    embedding_dir: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "csv_path": str(dataset_path),
            "embedding_dir": str(embedding_dir),
            "results_dir": str(tmp_path / "prototype_single_results"),
        },
        "features": {
            "normalize": True,
            "pooling_type": "mean",
        },
        "model": {
            "type": "simple",
            "input_dim": 8,
            "output_dim": 1,
            "dropout_rate": 0.1,
            "contrastive_dim": 8,
            "encoder_hidden_dim": 8,
        },
        "training": {
            "batch_size": 2,
            "num_folds": 2,
            "threshold_method": "youden",
            "target_recall": 0.8,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "early_stopping_patience": 1,
            "grad_clip_value": 1.0,
            "use_variants": False,
            "contrastive_type": "prototype_ranking",
            "majority_class": 0,
            "monitor_metric": "auprc",
            "mode": "max",
            "prototype_loss_params": {
                "temperature": 0.07,
                "eps_pos": 0.15,
                "eps_neg": 0.03,
                "bce_weight": 1.0,
                "scoring_temperature": 0.1,
                "unsupervised_weight": 0.0,
                "prototype_weight": 1.0,
                "prototype_update_strategy": "fixed",
                "prototype_momentum": 0.9,
            },
            "lr_scheduler": {
                "scheduler_type": "plateau",
                "patience": 1,
            },
        },
        "output": {
            "save_checkpoints": True,
            "plot_training_curves": False,
        },
        "hardware": {
            "gpu_id": -1,
            "random_seed": 42,
            "deterministic": True,
            "debug_logging": False,
            "num_workers": 0,
        },
    }


def make_prototype_two_stage_config(
    tmp_path: Path,
    dataset_path: Path,
    embedding_dir: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "pretraining_csv_path": str(dataset_path),
            "finetuning_csv_path": str(dataset_path),
            "embedding_dir": str(embedding_dir),
            "results_dir": str(tmp_path / "prototype_two_stage_results"),
            "label_config": {
                "label_column": "label",
                "sequence_id_column": "sequence_id",
            },
        },
        "features": {
            "normalize": True,
            "pooling_type": "mean",
        },
        "model": {
            "type": "simple",
            "input_dim": 8,
            "output_dim": 1,
            "dropout_rate": 0.1,
            "contrastive_dim": 8,
            "encoder_hidden_dim": 8,
        },
        "training": {
            "batch_size": 2,
            "num_folds": 2,
            "threshold_method": "youden",
            "target_recall": 0.8,
            "use_variants": False,
            "pretraining": {
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_epochs": 0,
                "early_stopping_patience": 1,
                "grad_clip_value": 1.0,
                "contrastive_type": "prototype",
                "majority_class": 0,
                "monitor_metric": "loss",
                "mode": "min",
                "prototype_loss_params": {
                    "temperature": 0.07,
                    "eps_pos": 0.15,
                    "eps_neg": 0.03,
                    "prototype_weight": 1.0,
                    "unsupervised_weight": 0.0,
                    "prototype_update_strategy": "fixed",
                    "prototype_momentum": 0.9,
                },
                "lr_scheduler": {
                    "scheduler_type": "plateau",
                    "patience": 1,
                },
            },
            "finetuning": {
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_epochs": 0,
                "early_stopping_patience": 1,
                "grad_clip_value": 1.0,
                "contrastive_type": "prototype_ranking",
                "majority_class": 0,
                "monitor_metric": "auprc",
                "mode": "max",
                "prototype_loss_params": {
                    "temperature": 0.07,
                    "eps_pos": 0.15,
                    "eps_neg": 0.03,
                    "bce_weight": 1.0,
                    "scoring_temperature": 0.1,
                    "unsupervised_weight": 0.0,
                    "prototype_weight": 1.0,
                    "prototype_update_strategy": "fixed",
                    "prototype_momentum": 0.9,
                },
                "lr_scheduler": {
                    "scheduler_type": "plateau",
                    "patience": 1,
                },
            },
        },
        "output": {
            "save_checkpoints": True,
            "plot_training_curves": False,
        },
        "hardware": {
            "gpu_id": -1,
            "random_seed": 42,
            "deterministic": True,
            "debug_logging": False,
            "num_workers": 0,
        },
    }


def merge_nested_dict(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_nested_dict(base[key], value)
        else:
            base[key] = value


def config_dict(config: dict[str, Any]) -> ConfigDict:
    return ConfigDict(config)
