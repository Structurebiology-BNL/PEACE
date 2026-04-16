from effector_bincls.training.baseline import main as baseline_main
from effector_bincls.training.prototype_single import main as prototype_single_main
from effector_bincls.training.prototype_two_stage import (
    main as prototype_two_stage_main,
)


def test_baseline_training_entrypoint_exports_main() -> None:
    assert callable(baseline_main)


def test_single_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_single_main)


def test_two_stage_training_entrypoint_exports_main() -> None:
    assert callable(prototype_two_stage_main)
