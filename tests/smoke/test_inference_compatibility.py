from __future__ import annotations

import csv
import sys

import numpy as np
import pytest
import torch

from effector_bincls.data import open_packed_embedding_dataset
from effector_bincls.inference.prototype import (
    load_embedding_batch,
)
from effector_bincls.inference.prototype import (
    main as prototype_inference_main,
)

from .conftest import create_historical_run_dir


@pytest.mark.smoke
def test_infer_prototype_accepts_old_single_stage_run_dir(
    monkeypatch, tmp_path
) -> None:
    run_dir, packed_embedding_dir = create_historical_run_dir(
        tmp_path,
        is_single_stage=True,
        prototype_in_extra_state=False,
    )
    output_file = tmp_path / "single_stage_predictions.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer-prototype",
            "--embedding_dir",
            str(packed_embedding_dir),
            "--model_dir",
            str(run_dir),
            "--output_file",
            str(output_file),
            "--single-stage",
            "--threshold",
            "0.5",
        ],
    )
    prototype_inference_main()

    with output_file.open() as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["sequence_id", "probability", "binary_prediction", "threshold"]
    assert len(rows) == 7


@pytest.mark.smoke
def test_inference_uses_original_variant_metadata_for_non_variant_path(
    tmp_path,
) -> None:
    _, packed_embedding_dir = create_historical_run_dir(
        tmp_path,
        is_single_stage=True,
    )
    packed_embeddings, _, metadata = open_packed_embedding_dataset(packed_embedding_dir)

    embedding_batch = load_embedding_batch(
        packed_embeddings[:1],
        normalize=False,
        use_variants=False,
        original_variant_index=int(metadata["original_variant_index"]),
    )

    assert torch.equal(
        embedding_batch,
        torch.tensor(np.asarray([[1.0] * 8], dtype=np.float32)),
    )


@pytest.mark.smoke
def test_infer_prototype_accepts_old_two_stage_run_dir(monkeypatch, tmp_path) -> None:
    run_dir, packed_embedding_dir = create_historical_run_dir(
        tmp_path,
        is_single_stage=False,
        prototype_in_extra_state=True,
    )
    output_file = tmp_path / "two_stage_predictions.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer-prototype",
            "--embedding_dir",
            str(packed_embedding_dir),
            "--model_dir",
            str(run_dir),
            "--output_file",
            str(output_file),
            "--threshold",
            "0.5",
        ],
    )
    prototype_inference_main()

    with output_file.open() as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["sequence_id", "probability", "binary_prediction", "threshold"]
    assert len(rows) == 7
