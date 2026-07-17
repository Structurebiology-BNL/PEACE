from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from ml_collections import ConfigDict

from effector_bincls.data import write_packed_embedding_dataset
from effector_bincls.training.baseline import main as baseline_main
from effector_bincls.training.contrastive_bce import (
    main as contrastive_bce_main,
)
from effector_bincls.training.contrastive_bce import (
    validate_contrastive_bce_config,
)
from effector_bincls.training.data import create_contrastive_bce_data_loader_fn
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
                "num_epochs": 1,
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
            },
            "output": {
                "save_checkpoints": False,
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


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "type"), "simple", "model.type='simple_predictor'"),
        (("model", "output_dim"), 2, "model.output_dim=1"),
        (("model", "output_dim"), 1.0, "model.output_dim=1"),
        (("model", "use_contrastive"), False, "use_contrastive=true"),
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
    ],
)
def test_validate_contrastive_bce_config_rejects_invalid_values(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw_config = _valid_contrastive_bce_config().to_dict()
    _set_nested(raw_config, path, value)
    config = ConfigDict(raw_config)

    with pytest.raises(ValueError, match=message):
        validate_contrastive_bce_config(config)


def test_validate_contrastive_bce_config_accepts_valid_config() -> None:
    validate_contrastive_bce_config(_valid_contrastive_bce_config())


def test_invalid_config_does_not_create_results_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _valid_contrastive_bce_config(tmp_path)
    config.training.temperature = 0.0
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    results_dir = Path(config.data.results_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-contrastive-bce", "--config", str(config_path)],
    )

    with pytest.raises(ValueError, match="training.temperature"):
        contrastive_bce_main()

    assert not results_dir.exists()


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


def test_contrastive_bce_training_entrypoint_exports_main() -> None:
    assert callable(contrastive_bce_main)


def test_single_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_single_main)


def test_two_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_two_stage_main)
