import pytest
import torch
from ml_collections import ConfigDict

from effector_bincls.evaluation.baseline import load_baseline_model
from effector_bincls.evaluation.prototype import load_prototype_ranking_model
from effector_bincls.models import SimplePredictor


def make_prototype_model() -> SimplePredictor:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=4,
        encoder_hidden_dim=8,
    )
    model.set_training_mode("pretraining")
    return model


def test_package_baseline_checkpoint_loader_accepts_current_simple_only_checkpoint(
    tmp_path,
) -> None:
    config = ConfigDict(
        {
            "model": {
                "type": "simple_predictor",
                "input_dim": 8,
                "output_dim": 1,
                "dropout_rate": 0.1,
                "use_contrastive": False,
                "encoder_hidden_dim": 8,
            }
        }
    )
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=False,
        encoder_hidden_dim=8,
    )
    checkpoint_path = tmp_path / "baseline_checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    loaded_model = load_baseline_model(checkpoint_path, config, torch.device("cpu"))

    assert isinstance(loaded_model, SimplePredictor)
    assert loaded_model.training_mode == "finetuning"


def test_package_prototype_loader_accepts_top_level_prototypes(tmp_path) -> None:
    config = ConfigDict(
        {
            "model": {
                "type": "simple",
                "input_dim": 8,
                "output_dim": 1,
                "dropout_rate": 0.1,
                "contrastive_dim": 4,
                "encoder_hidden_dim": 8,
            },
            "training": {
                "prototype_loss_params": {
                    "scoring_temperature": 0.7,
                }
            },
        }
    )
    checkpoint_path = tmp_path / "prototype_checkpoint.pt"
    torch.save(
        {
            "model_state": make_prototype_model().state_dict(),
            "prototypes": torch.randn(2, 4),
        },
        checkpoint_path,
    )

    loaded_model, prototypes, scoring_temperature = load_prototype_ranking_model(
        checkpoint_path,
        config,
        torch.device("cpu"),
        is_single_stage=True,
    )

    assert isinstance(loaded_model, SimplePredictor)
    assert loaded_model.training_mode == "pretraining"
    assert prototypes is not None
    assert prototypes.shape == (2, 4)
    assert scoring_temperature == pytest.approx(0.7)


def test_package_prototype_loader_accepts_extra_state_prototypes(tmp_path) -> None:
    config = ConfigDict(
        {
            "model": {
                "type": "simple",
                "input_dim": 8,
                "output_dim": 1,
                "dropout_rate": 0.1,
                "contrastive_dim": 4,
                "encoder_hidden_dim": 8,
            },
            "training": {
                "prototype_loss_params": {
                    "scoring_temperature": 0.9,
                }
            },
        }
    )
    checkpoint_path = tmp_path / "prototype_extra_state_checkpoint.pt"
    torch.save(
        {
            "model_state": make_prototype_model().state_dict(),
            "extra_state": {
                "prototypes": torch.randn(2, 4),
            },
        },
        checkpoint_path,
    )

    _, prototypes, scoring_temperature = load_prototype_ranking_model(
        checkpoint_path,
        config,
        torch.device("cpu"),
        is_single_stage=True,
    )

    assert prototypes is not None
    assert prototypes.shape == (2, 4)
    assert scoring_temperature == pytest.approx(0.9)


def test_package_prototype_loader_reads_two_stage_hybrid_temperature(
    tmp_path,
) -> None:
    config = ConfigDict(
        {
            "model": {
                "type": "simple",
                "input_dim": 8,
                "output_dim": 1,
                "dropout_rate": 0.1,
                "contrastive_dim": 4,
                "encoder_hidden_dim": 8,
            },
            "training": {
                "finetuning": {
                    "hybrid_loss_params": {
                        "scoring_temperature": 0.35,
                    }
                }
            },
        }
    )
    checkpoint_path = tmp_path / "prototype_two_stage_checkpoint.pt"
    torch.save(
        {
            "model_state": make_prototype_model().state_dict(),
            "prototypes": torch.randn(2, 4),
        },
        checkpoint_path,
    )

    _, prototypes, scoring_temperature = load_prototype_ranking_model(
        checkpoint_path,
        config,
        torch.device("cpu"),
        is_single_stage=False,
    )

    assert prototypes is not None
    assert scoring_temperature == pytest.approx(0.35)
