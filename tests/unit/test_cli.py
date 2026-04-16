from effector_bincls.cli import main as cli_main


def test_cli_exports_main() -> None:
    assert callable(cli_main)
