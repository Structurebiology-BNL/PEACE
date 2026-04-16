from pathlib import Path

import pytest
from ml_collections import ConfigDict

from effector_bincls.training.validation import (
    validate_baseline_training_config,
    validate_prototype_single_stage_config,
    validate_prototype_two_stage_config,
)


def make_config(model_type: str) -> ConfigDict:
    return ConfigDict(
        {
            "model": {
                "type": model_type,
                "input_dim": 1024,
                "output_dim": 1,
                "dropout_rate": 0.2,
            },
            "training": {
                "num_folds": 2,
            },
            "data": {},
        }
    )


def test_validate_baseline_training_config_accepts_simple_predictor() -> None:
    config = make_config("simple_predictor")

    validate_baseline_training_config(config)


def test_validate_baseline_training_config_rejects_wrong_model_type() -> None:
    config = make_config("simple")

    with pytest.raises(ValueError, match="model.type='simple_predictor'"):
        validate_baseline_training_config(config)


def test_validate_prototype_single_stage_config_rejects_two_stage_sections() -> None:
    config = make_config("simple")
    config.training.contrastive_type = "prototype_ranking"
    config.training.pretraining = ConfigDict({"num_epochs": 1})

    with pytest.raises(ValueError, match="single-stage configuration"):
        validate_prototype_single_stage_config(config)


def test_validate_prototype_two_stage_config_requires_existing_datasets(
    tmp_path: Path,
) -> None:
    config = make_config("simple")
    config.training.pretraining = ConfigDict({"contrastive_type": "prototype"})
    config.training.finetuning = ConfigDict({"contrastive_type": "prototype_ranking"})
    config.data.pretraining_csv_path = str(tmp_path / "missing_pretraining.csv")
    config.data.finetuning_csv_path = str(tmp_path / "missing_finetuning.csv")

    with pytest.raises(FileNotFoundError, match="Pretraining dataset not found"):
        validate_prototype_two_stage_config(config)


def test_validate_prototype_two_stage_config_accepts_existing_datasets(
    tmp_path: Path,
) -> None:
    pretraining_csv = tmp_path / "pretraining.csv"
    finetuning_csv = tmp_path / "finetuning.csv"
    pretraining_csv.write_text("sequence_id,label,partition\np1,1,train\np2,0,train\n")
    finetuning_csv.write_text("sequence_id,label,partition\np1,1,train\np2,0,test\n")

    config = make_config("simple")
    config.training.pretraining = ConfigDict({"contrastive_type": "prototype"})
    config.training.finetuning = ConfigDict({"contrastive_type": "prototype_ranking"})
    config.data.pretraining_csv_path = str(pretraining_csv)
    config.data.finetuning_csv_path = str(finetuning_csv)

    validate_prototype_two_stage_config(config)
