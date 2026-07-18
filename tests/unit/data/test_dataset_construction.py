from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.dataset_construction.combine_pos_and_neg_csv import (
    build_pretraining_runtime_dataset,
)


def test_build_pretraining_runtime_dataset_materializes_unique_union() -> None:
    combined = pd.DataFrame(
        [
            ("pretrain-only", "AAA", 1, "pretrain", 10.0),
            ("shared", "BBB", 1, "pretrain", 11.0),
            ("shared", "BBB", 1, "train", 11.0),
            ("train-negative", "CCC", 0, "train", None),
            ("held-out", "DDD", 1, "test", 12.0),
        ],
        columns=["sequence_id", "sequence", "label", "partition", "cluster_id"],
    )

    result = build_pretraining_runtime_dataset(combined)

    expected = pd.DataFrame(
        [
            ("pretrain-only", "AAA", 1, "train", 10.0),
            ("shared", "BBB", 1, "train", 11.0),
            ("train-negative", "CCC", 0, "train", None),
        ],
        columns=combined.columns,
    )
    assert_frame_equal(result.reset_index(drop=True), expected)
    assert result["sequence_id"].is_unique


@pytest.mark.parametrize("sequence_id", [None, "", "   "])
def test_build_pretraining_runtime_dataset_rejects_null_or_blank_sequence_ids(
    sequence_id: object,
) -> None:
    combined = pd.DataFrame(
        [(sequence_id, "AAA", 1, "train")],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(ValueError, match="sequence IDs.*null or blank.*positions"):
        build_pretraining_runtime_dataset(combined)


@pytest.mark.parametrize("partition", [None, "", "   ", "validation"])
def test_build_pretraining_runtime_dataset_rejects_invalid_partitions(
    partition: object,
) -> None:
    combined = pd.DataFrame(
        [("protein-1", "AAA", 1, partition)],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(
        ValueError,
        match="partition.*only.*train.*pretrain.*test",
    ):
        build_pretraining_runtime_dataset(combined)


def test_build_pretraining_runtime_dataset_bounds_invalid_partition_examples() -> None:
    combined = pd.DataFrame(
        [(f"protein-{index}", "AAA", 1, f"invalid-{index}") for index in range(7)],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(ValueError) as exc_info:
        build_pretraining_runtime_dataset(combined)

    message = str(exc_info.value)
    assert "invalid-0" in message
    assert "invalid-4" in message
    assert "invalid-5" not in message
    assert "invalid-6" not in message


@pytest.mark.parametrize(
    ("changed_column", "changed_value"),
    [("sequence", "DIFFERENT"), ("label", 0), ("cluster_id", 99.0)],
)
def test_build_pretraining_runtime_dataset_rejects_conflicting_identity_rows(
    changed_column: str,
    changed_value: object,
) -> None:
    combined = pd.DataFrame(
        [
            ("shared", "BBB", 1, "pretrain", 11.0),
            ("shared", "BBB", 1, "train", 11.0),
        ],
        columns=["sequence_id", "sequence", "label", "partition", "cluster_id"],
    )
    combined.loc[1, changed_column] = changed_value

    with pytest.raises(ValueError, match=f"shared.*{changed_column}"):
        build_pretraining_runtime_dataset(combined)


@pytest.mark.parametrize(
    "memberships",
    [
        ["train", "train"],
        ["train", "train", "pretrain"],
    ],
)
def test_build_pretraining_runtime_dataset_rejects_duplicate_membership(
    memberships: list[str],
) -> None:
    combined = pd.DataFrame(
        [("shared", "BBB", 1, membership) for membership in memberships],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(ValueError, match="shared.*train.*pretrain"):
        build_pretraining_runtime_dataset(combined)
