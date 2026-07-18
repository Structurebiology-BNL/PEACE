from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ml_collections import ConfigDict

from effector_bincls.data import write_packed_embedding_dataset
from effector_bincls.run_utils import load_config
from effector_bincls.training.baseline import main as baseline_main
from effector_bincls.training.contrastive_bce import (
    main as contrastive_bce_main,
)
from effector_bincls.training.contrastive_bce import (
    validate_contrastive_bce_config,
)
from effector_bincls.training.data import (
    create_contrastive_bce_data_loader_fn,
    load_test_data,
)
from effector_bincls.training.prototype_single import main as prototype_single_main
from effector_bincls.training.prototype_two_stage import (
    main as prototype_two_stage_main,
)


def test_baseline_training_entrypoint_exports_main() -> None:
    assert callable(baseline_main)


def _valid_contrastive_bce_config(tmp_path: Path | None = None) -> ConfigDict:
    root = tmp_path or Path("/tmp/contrastive-bce-test")
    return ConfigDict(
        {
            "data": {
                "csv_path": str(root / "dataset.csv"),
                "embedding_dir": str(root / "embeddings"),
                "results_dir": str(root / "results"),
            },
            "features": {
                "normalize": False,
                "pooling_type": "mean",
            },
            "model": {
                "type": "simple_predictor",
                "input_dim": 3,
                "output_dim": 1,
                "dropout_rate": 0.0,
                "use_contrastive": True,
                "contrastive_dim": 2,
                "encoder_hidden_dim": 4,
            },
            "training": {
                "batch_size": 2,
                "num_folds": 2,
                "threshold_method": "youden",
                "target_recall": 0.85,
                "num_epochs": 2,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "grad_clip_value": 2.0,
                "loss_type": "contrastive_bce",
                "bce_weight": 1.0,
                "unsupervised_weight": 0.5,
                "temperature": 0.07,
                "use_variants": True,
                "variant_sampling": {
                    "enabled": True,
                    "num_variants": 2,
                    "always_include_original": True,
                },
                "monitor_metric": "auprc",
                "mode": "max",
                "lr_scheduler": {
                    "scheduler_type": "cosine",
                    "eta_min": 1e-7,
                },
            },
            "output": {
                "save_checkpoints": False,
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
    )


def _set_nested(config: dict, path: tuple[str, ...], value: object) -> None:
    section = config
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value


def _assert_entrypoint_rejects_without_results(
    monkeypatch,
    tmp_path: Path,
    config: ConfigDict,
    message: str,
    *,
    results_dir: Path | None = None,
) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    expected_results_dir = results_dir or Path(config.data.results_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-contrastive-bce", "--config", str(config_path)],
    )

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        contrastive_bce_main()

    assert not expected_results_dir.exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("features", "normalize"), "false", "features.normalize must be boolean"),
        (("features", "pooling_type"), "", "features.pooling_type must be non-empty"),
        (("model", "type"), "simple", "model.type='simple_predictor'"),
        (("model", "output_dim"), 2, "model.output_dim=1"),
        (("model", "output_dim"), 1.0, "model.output_dim=1"),
        (("model", "use_contrastive"), False, "use_contrastive=true"),
        (
            ("model", "dropout_rate"),
            1.0,
            r"model.dropout_rate must be in \[0, 1\)",
        ),
        (("model", "input_dim"), 0, "model.input_dim must be a positive integer"),
        (
            ("model", "encoder_hidden_dim"),
            -1,
            "model.encoder_hidden_dim must be a positive integer",
        ),
        (
            ("model", "contrastive_dim"),
            0,
            "model.contrastive_dim must be a positive integer",
        ),
        (
            ("training", "batch_size"),
            0,
            "training.batch_size must be a positive integer",
        ),
        (("training", "num_folds"), 1, "training.num_folds must be at least 2"),
        (
            ("training", "num_epochs"),
            0,
            "training.num_epochs must be a positive integer",
        ),
        (
            ("training", "learning_rate"),
            0.0,
            "training.learning_rate must be finite and > 0",
        ),
        (
            ("training", "weight_decay"),
            -0.1,
            "training.weight_decay must be finite and >= 0",
        ),
        (
            ("training", "warmup_epochs"),
            3,
            "training.warmup_epochs must not exceed training.num_epochs",
        ),
        (
            ("training", "early_stopping_patience"),
            0,
            "training.early_stopping_patience must be a positive integer",
        ),
        (
            ("training", "grad_clip_value"),
            float("inf"),
            "training.grad_clip_value must be finite and > 0",
        ),
        (
            ("training", "threshold_method"),
            "accuracy",
            "training.threshold_method must be one of",
        ),
        (
            ("training", "target_recall"),
            1.1,
            r"training.target_recall must be in \(0, 1\]",
        ),
        (("training", "loss_type"), "bce", "loss_type='contrastive_bce'"),
        (
            ("training", "temperature"),
            0.0,
            "training.temperature must be finite and > 0",
        ),
        (
            ("training", "temperature"),
            float("nan"),
            "training.temperature must be finite and > 0",
        ),
        (("training", "bce_weight"), 0.0, "training.bce_weight must be finite and > 0"),
        (
            ("training", "unsupervised_weight"),
            float("inf"),
            "training.unsupervised_weight must be finite and > 0",
        ),
        (("training", "use_variants"), False, "training.use_variants=true"),
        (
            ("training", "variant_sampling", "enabled"),
            False,
            "variant_sampling.enabled=true",
        ),
        (
            ("training", "variant_sampling", "num_variants"),
            1,
            "variant_sampling.num_variants must be at least 2",
        ),
        (
            ("training", "variant_sampling", "always_include_original"),
            False,
            "always_include_original=true",
        ),
        (
            ("training", "monitor_metric"),
            "accuracy",
            "training.monitor_metric must be one of",
        ),
        (
            ("training", "mode"),
            "min",
            "training.mode='max'.*monitor_metric='auprc'",
        ),
        (
            ("training", "lr_scheduler", "scheduler_type"),
            "linear",
            "scheduler_type must be one of",
        ),
        (
            ("training", "lr_scheduler", "eta_min"),
            -1.0,
            "eta_min must be finite and >= 0",
        ),
        (
            ("output", "save_checkpoints"),
            1,
            "output.save_checkpoints must be boolean",
        ),
        (
            ("hardware", "gpu_id"),
            -2,
            "hardware.gpu_id must be an integer >= -1",
        ),
        (
            ("hardware", "random_seed"),
            -1,
            "hardware.random_seed must be an integer >= 0",
        ),
        (
            ("hardware", "num_workers"),
            -1,
            "hardware.num_workers must be an integer >= 0",
        ),
    ],
)
def test_contrastive_bce_entrypoint_rejects_invalid_config_before_setup(
    monkeypatch,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    _set_nested(raw_config, path, value)
    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        ConfigDict(raw_config),
        message,
    )


def test_validate_contrastive_bce_config_accepts_valid_config() -> None:
    validate_contrastive_bce_config(_valid_contrastive_bce_config())


def test_validate_contrastive_bce_config_accepts_shipped_public_config() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    public_config = ConfigDict(
        load_config(repo_root / "src/configs/contrastive_bce.yaml")
    )

    validate_contrastive_bce_config(public_config)


@pytest.mark.parametrize(
    "section",
    ["data", "features", "model", "training", "output", "hardware"],
)
def test_contrastive_bce_entrypoint_requires_public_sections_before_setup(
    monkeypatch,
    tmp_path: Path,
    section: str,
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    results_dir = Path(raw_config["data"]["results_dir"])
    del raw_config[section]
    config_path = tmp_path / f"missing-{section}.yml"
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-contrastive-bce", "--config", str(config_path)],
    )

    with pytest.raises(ValueError, match=f"requires a {section} section"):
        contrastive_bce_main()

    assert not results_dir.exists()


def _delete_nested(config: dict, path: tuple[str, ...]) -> None:
    section = config
    for key in path[:-1]:
        section = section[key]
    del section[path[-1]]


@pytest.mark.parametrize(
    "path",
    [
        ("data", "csv_path"),
        ("data", "embedding_dir"),
        ("data", "results_dir"),
        ("features", "normalize"),
        ("features", "pooling_type"),
        ("model", "type"),
        ("model", "input_dim"),
        ("model", "output_dim"),
        ("model", "dropout_rate"),
        ("model", "use_contrastive"),
        ("model", "contrastive_dim"),
        ("model", "encoder_hidden_dim"),
        ("training", "batch_size"),
        ("training", "num_folds"),
        ("training", "threshold_method"),
        ("training", "target_recall"),
        ("training", "num_epochs"),
        ("training", "learning_rate"),
        ("training", "weight_decay"),
        ("training", "warmup_epochs"),
        ("training", "early_stopping_patience"),
        ("training", "grad_clip_value"),
        ("training", "use_variants"),
        ("training", "variant_sampling"),
        ("training", "loss_type"),
        ("training", "bce_weight"),
        ("training", "unsupervised_weight"),
        ("training", "temperature"),
        ("training", "monitor_metric"),
        ("training", "mode"),
        ("training", "lr_scheduler"),
        ("output", "save_checkpoints"),
        ("output", "plot_training_curves"),
        ("hardware", "gpu_id"),
        ("hardware", "random_seed"),
        ("hardware", "deterministic"),
        ("hardware", "debug_logging"),
        ("hardware", "num_workers"),
    ],
)
def test_contrastive_bce_entrypoint_rejects_missing_fields_before_setup(
    monkeypatch,
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    results_dir = Path(raw_config["data"]["results_dir"])
    _delete_nested(raw_config, path)

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        ConfigDict(raw_config),
        path[-1],
        results_dir=results_dir,
    )


def _write_toy_dataset(tmp_path: Path, *, num_variants: int = 2) -> ConfigDict:
    config = _valid_contrastive_bce_config(tmp_path)
    Path(config.data.csv_path).write_text(
        "sequence_id,label,partition\n"
        "seq0,1,train\n"
        "seq1,0,train\n"
        "seq2,1,train\n"
        "seq3,0,train\n"
        "seq4,1,test\n"
        "seq5,0,test\n"
    )
    embeddings = np.arange(
        6 * num_variants * 3,
        dtype=np.float32,
    ).reshape(6, num_variants, 3)
    write_packed_embedding_dataset(
        config.data.embedding_dir,
        [f"seq{index}" for index in range(6)],
        embeddings,
        pooling_type="mean",
        original_variant_index=0,
    )
    return config


def test_contrastive_bce_loader_returns_multi_view_batches(tmp_path: Path) -> None:
    config = _write_toy_dataset(tmp_path)

    train_loader, val_loader = create_contrastive_bce_data_loader_fn(config)(1)
    train_features, _ = next(iter(train_loader))
    val_features, _ = next(iter(val_loader))

    assert train_features.ndim == 3
    assert train_features.shape[1:] == (2, 3)
    assert val_features.ndim == 3
    assert val_features.shape[1:] == (2, 3)


def test_contrastive_bce_loader_rejects_too_few_packed_views(
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path, num_variants=1)

    with pytest.raises(ValueError, match="contains 1 variants.*requests 2"):
        create_contrastive_bce_data_loader_fn(config)


def test_contrastive_bce_test_loader_places_canonical_view_first(
    tmp_path: Path,
) -> None:
    config = _valid_contrastive_bce_config(tmp_path)
    Path(config.data.csv_path).write_text(
        "sequence_id,label,partition\n"
        "seq0,1,train\n"
        "seq1,0,train\n"
        "seq2,1,test\n"
        "seq3,0,test\n"
    )
    embeddings = np.asarray(
        [
            [[100.0, 100.0, 100.0], [1.0, 1.0, 1.0]],
            [[101.0, 101.0, 101.0], [2.0, 2.0, 2.0]],
            [[102.0, 102.0, 102.0], [3.0, 3.0, 3.0]],
            [[103.0, 103.0, 103.0], [4.0, 4.0, 4.0]],
        ],
        dtype=np.float32,
    )
    write_packed_embedding_dataset(
        config.data.embedding_dir,
        [f"seq{index}" for index in range(4)],
        embeddings,
        pooling_type="mean",
        original_variant_index=1,
    )

    features, _ = next(iter(load_test_data(config)))

    assert torch.equal(features[:, 0, :], torch.tensor([[3.0] * 3, [4.0] * 3]))


def test_contrastive_bce_training_entrypoint_exports_main() -> None:
    assert callable(contrastive_bce_main)


def test_single_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_single_main)


def test_two_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_two_stage_main)
