from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict

from effector_bincls.data import (
    SimpleDataset,
    load_labeled_dataset,
    validate_two_stage_dataset_pair,
)
from effector_bincls.data.packed_embeddings import write_packed_embedding_dataset
from effector_bincls.training.data import create_variant_collate_fn, load_test_data


def _write_dataset_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("sequence_id,label,partition\n" + "\n".join(rows) + "\n")
    return path


def _write_packed_embeddings(
    path: Path,
    sequence_ids: list[str],
    embeddings: np.ndarray,
    *,
    original_variant_index: int = 0,
) -> Path:
    return write_packed_embedding_dataset(
        path,
        sequence_ids,
        embeddings,
        pooling_type="mean",
        original_variant_index=original_variant_index,
    )


def test_load_labeled_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("sequence_id,label\nseq0,1\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_labeled_dataset(csv_path, required_partitions={"train"})


def test_load_labeled_dataset_rejects_missing_required_partitions(
    tmp_path: Path,
) -> None:
    csv_path = _write_dataset_csv(tmp_path / "dataset.csv", ["seq0,1,train"])

    with pytest.raises(ValueError, match="Missing required partitions"):
        load_labeled_dataset(csv_path, required_partitions={"train", "test"})


def test_validate_two_stage_dataset_pair_rejects_missing_samples_and_label_mismatches(
    tmp_path: Path,
) -> None:
    pretraining_csv = _write_dataset_csv(
        tmp_path / "pretraining.csv",
        [
            "seq0,1,train",
            "seq1,0,train",
            "seq2,1,train",
        ],
    )
    finetuning_csv = _write_dataset_csv(
        tmp_path / "finetuning.csv",
        [
            "seq0,1,train",
            "seq1,1,test",
            "seq9,0,test",
        ],
    )

    pretraining_df = load_labeled_dataset(
        pretraining_csv,
        required_partitions={"train"},
    )
    finetuning_df = load_labeled_dataset(
        finetuning_csv,
        required_partitions={"train", "test"},
    )

    with pytest.raises(ValueError, match="samples not in pretraining dataset"):
        validate_two_stage_dataset_pair(
            pretraining_df,
            finetuning_df,
            pretraining_csv_path=pretraining_csv,
            finetuning_csv_path=finetuning_csv,
        )

    finetuning_csv = _write_dataset_csv(
        tmp_path / "finetuning.csv",
        [
            "seq0,1,train",
            "seq1,1,test",
        ],
    )
    finetuning_df = load_labeled_dataset(
        finetuning_csv,
        required_partitions={"train", "test"},
    )

    with pytest.raises(ValueError, match="Label inconsistencies detected"):
        validate_two_stage_dataset_pair(
            pretraining_df,
            finetuning_df,
            pretraining_csv_path=pretraining_csv,
            finetuning_csv_path=finetuning_csv,
        )


def test_simple_dataset_rejects_missing_embeddings(tmp_path: Path) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        [
            "seq0,1,train",
            "seq1,0,test",
        ],
    )
    embedding_dir = tmp_path / "embeddings"
    _write_packed_embeddings(
        embedding_dir,
        ["seq0"],
        np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
    )

    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        SimpleDataset(
            embedding_dir=embedding_dir,
            csv_path=csv_path,
            sequence_ids=["seq0", "seq1"],
        )


def test_simple_dataset_uses_original_variant_metadata_for_baseline_path(
    tmp_path: Path,
) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        [
            "seq0,1,train",
            "seq1,0,test",
        ],
    )
    embedding_dir = _write_packed_embeddings(
        tmp_path / "embeddings",
        ["seq0", "seq1"],
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype=np.float32,
        ),
        original_variant_index=1,
    )

    dataset = SimpleDataset(
        embedding_dir=embedding_dir,
        csv_path=csv_path,
        sequence_ids=["seq0"],
        normalize=False,
        use_variants=False,
    )

    embedding, label = dataset[0]

    assert embedding.shape == (3,)
    assert torch.equal(embedding, torch.tensor([10.0, 20.0, 30.0]))
    assert torch.equal(label, torch.tensor([1.0]))


def test_simple_dataset_returns_all_variants_when_requested(tmp_path: Path) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        [
            "seq0,1,train",
        ],
    )
    embedding_dir = _write_packed_embeddings(
        tmp_path / "embeddings",
        ["seq0"],
        np.asarray(
            [
                [[3.0, 0.0, 4.0], [5.0, 12.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )

    dataset = SimpleDataset(
        embedding_dir=embedding_dir,
        csv_path=csv_path,
        sequence_ids=["seq0"],
        normalize=True,
        use_variants=True,
    )

    embedding, _ = dataset[0]

    assert embedding.shape == (2, 3)
    assert torch.allclose(
        embedding,
        torch.tensor(
            [
                [0.6, 0.0, 0.8],
                [5.0 / 13.0, 12.0 / 13.0, 0.0],
            ]
        ),
    )


def test_load_test_data_rejects_missing_test_embeddings(tmp_path: Path) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        [
            "seq0,1,train",
            "seq1,0,test",
        ],
    )
    embedding_dir = tmp_path / "embeddings"
    _write_packed_embeddings(
        embedding_dir,
        ["seq0"],
        np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
    )

    config = ConfigDict(
        {
            "data": {
                "csv_path": str(csv_path),
                "embedding_dir": str(embedding_dir),
            },
            "features": {
                "normalize": True,
                "pooling_type": "mean",
            },
            "training": {
                "batch_size": 2,
                "use_variants": False,
            },
            "hardware": {
                "num_workers": 0,
            },
            "model": {
                "type": "simple_predictor",
            },
        }
    )

    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        load_test_data(config, test_csv_path=csv_path)


def test_variant_collate_fn_includes_original_variant_index_when_sampling() -> None:
    collate_fn = create_variant_collate_fn(
        {
            "enabled": True,
            "num_variants": 2,
            "always_include_original": True,
        },
        random_seed=123,
        original_variant_index=1,
    )

    embeddings, labels = collate_fn(
        [
            (
                torch.tensor(
                    [
                        [100.0, 100.0],
                        [1.0, 1.0],
                        [200.0, 200.0],
                    ]
                ),
                torch.tensor([1.0]),
            )
        ]
    )

    assert embeddings.shape == (1, 2, 2)
    assert torch.equal(embeddings[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(labels, torch.tensor([[1.0]]))
