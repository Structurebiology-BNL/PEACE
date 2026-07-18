import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ml_collections import ConfigDict

from effector_bincls.evaluation import baseline as baseline_evaluation
from effector_bincls.evaluation.prototype import main as prototype_evaluation_main
from effector_bincls.models import SimplePredictor


class StopAfterDataLoad(RuntimeError):
    pass


def test_baseline_evaluation_entrypoint_exports_main() -> None:
    assert callable(baseline_evaluation.main)


def test_prototype_evaluation_entrypoint_exports_main() -> None:
    assert callable(prototype_evaluation_main)


def test_baseline_evaluation_requests_canonical_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yml").write_text("model: {}\n")
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("sequence_id,label,partition\nseq0,1,test\n")
    args = SimpleNamespace(
        run_dir=tmp_path,
        test_csv=test_csv,
        threshold_method="youden",
        target_recall=0.85,
    )
    config = ConfigDict(
        {
            "model": {"type": "simple_predictor"},
            "training": {"use_variants": True},
        }
    )

    monkeypatch.setattr(baseline_evaluation, "parse_evaluation_args", lambda _: args)
    monkeypatch.setattr(baseline_evaluation, "load_run_config", lambda _: config)
    monkeypatch.setattr(
        baseline_evaluation,
        "setup_logger",
        lambda **_: logging.getLogger("baseline-canonical-test"),
    )
    monkeypatch.setattr(baseline_evaluation, "resolve_device", lambda _: "cpu")

    def stop_after_load(*args, **kwargs):
        assert kwargs["use_variants_override"] is False
        raise StopAfterDataLoad

    monkeypatch.setattr(baseline_evaluation, "load_test_data", stop_after_load)

    with pytest.raises(StopAfterDataLoad):
        baseline_evaluation.main()


def test_canonical_only_logits_match_canonical_first_multiview_logits() -> None:
    torch.manual_seed(7)
    model = SimplePredictor(
        input_dim=3,
        output_dim=1,
        dropout_rate=0.0,
        use_contrastive=True,
        contrastive_dim=2,
        encoder_hidden_dim=4,
    )
    model.eval()
    canonical = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    noncanonical = torch.tensor(
        [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
        dtype=torch.float32,
    )
    multiview = torch.stack([canonical, noncanonical], dim=1)

    with torch.no_grad():
        canonical_logits, _ = model(canonical)
        multiview_logits, _ = model(multiview)

    assert torch.equal(canonical_logits, multiview_logits)
    assert torch.equal(
        torch.sigmoid(canonical_logits),
        torch.sigmoid(multiview_logits),
    )
