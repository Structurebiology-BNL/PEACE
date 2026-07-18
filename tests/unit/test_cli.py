from effector_bincls.cli import COMMANDS
from effector_bincls.cli import main as cli_main


def test_cli_exports_main() -> None:
    assert callable(cli_main)


def test_cli_registers_contrastive_bce() -> None:
    assert "train-contrastive-bce" in COMMANDS
    assert callable(COMMANDS["train-contrastive-bce"][0])
