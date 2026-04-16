from effector_bincls.analysis.baseline import main as baseline_analysis_main
from effector_bincls.analysis.prototype import main as prototype_analysis_main


def test_baseline_analysis_entrypoint_exports_main() -> None:
    assert callable(baseline_analysis_main)


def test_prototype_analysis_entrypoint_exports_main() -> None:
    assert callable(prototype_analysis_main)
