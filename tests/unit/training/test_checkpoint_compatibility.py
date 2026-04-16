from pathlib import Path

from effector_bincls.checkpoints import get_checkpoint_path


def test_get_checkpoint_path_supports_historical_single_stage_layout() -> None:
    run_dir = Path("/tmp/example_run")

    assert get_checkpoint_path(run_dir, 3, True) == run_dir / "fold_3" / "checkpoint.pt"


def test_get_checkpoint_path_supports_historical_two_stage_layout() -> None:
    run_dir = Path("/tmp/example_run")

    assert get_checkpoint_path(run_dir, 3, False) == (
        run_dir / "fold_3" / "finetuning" / "checkpoint.pt"
    )
