from effector_bincls.inference.prototype import main as prototype_inference_main


def test_prototype_inference_entrypoint_exports_main() -> None:
    assert callable(prototype_inference_main)
