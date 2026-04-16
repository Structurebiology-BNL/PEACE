from effector_bincls.evaluation.baseline import main as baseline_evaluation_main
from effector_bincls.evaluation.prototype import main as prototype_evaluation_main


def test_baseline_evaluation_entrypoint_exports_main() -> None:
    assert callable(baseline_evaluation_main)


def test_prototype_evaluation_entrypoint_exports_main() -> None:
    assert callable(prototype_evaluation_main)
