from pathlib import Path

import torch
from ml_collections import ConfigDict

from effector_bincls.checkpoints import get_checkpoint_path, load_baseline_model
from effector_bincls.models import SimplePredictor


def test_get_checkpoint_path_supports_historical_single_stage_layout() -> None:
    run_dir = Path("/tmp/example_run")

    assert get_checkpoint_path(run_dir, 3, True) == run_dir / "fold_3" / "checkpoint.pt"


def test_get_checkpoint_path_supports_historical_two_stage_layout() -> None:
    run_dir = Path("/tmp/example_run")

    assert get_checkpoint_path(run_dir, 3, False) == (
        run_dir / "fold_3" / "finetuning" / "checkpoint.pt"
    )


def test_load_baseline_model_restores_configured_contrastive_dimension(
    tmp_path: Path,
) -> None:
    model = SimplePredictor(
        input_dim=5,
        output_dim=1,
        dropout_rate=0.0,
        use_contrastive=True,
        contrastive_dim=3,
        encoder_hidden_dim=4,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    config = ConfigDict(
        {
            "model": {
                "type": "simple_predictor",
                "input_dim": 5,
                "output_dim": 1,
                "dropout_rate": 0.0,
                "use_contrastive": True,
                "contrastive_dim": 3,
                "encoder_hidden_dim": 4,
            }
        }
    )

    loaded = load_baseline_model(checkpoint_path, config, torch.device("cpu"))

    assert loaded.contrastive_head[1].weight.shape == (3, 4)
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
