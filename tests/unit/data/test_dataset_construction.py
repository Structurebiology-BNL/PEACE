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


def test_build_pretraining_runtime_dataset_rejects_duplicate_membership() -> None:
    combined = pd.DataFrame(
        [
            ("shared", "BBB", 1, "train"),
            ("shared", "BBB", 1, "train"),
        ],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(ValueError, match="shared.*train.*pretrain"):
        build_pretraining_runtime_dataset(combined)
