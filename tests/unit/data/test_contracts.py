from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from effector_bincls.data import load_labeled_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "src" / "data"


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["sequence_id", "partition"]).reset_index(drop=True)


def _read_provenance_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sequence_id": str})


def _assert_valid_repeated_provenance_memberships(
    provenance_df: pd.DataFrame,
) -> None:
    repeated = provenance_df[provenance_df["sequence_id"].duplicated(keep=False)]
    identity_columns = [
        column for column in provenance_df.columns if column != "partition"
    ]
    for sequence_id, group in repeated.groupby("sequence_id"):
        assert len(group) == 2, (
            f"Repeated provenance sequence ID {sequence_id!r} must have exactly "
            "two rows."
        )
        membership_counts = group["partition"].value_counts().to_dict()
        assert membership_counts == {
            "train": 1,
            "pretrain": 1,
        }, (
            f"Repeated provenance sequence ID {sequence_id!r} must have exactly "
            "one train and one pretrain membership; "
            f"got {membership_counts}."
        )
        for column in identity_columns:
            assert group[column].nunique(dropna=False) == 1, (
                f"Repeated provenance sequence ID {sequence_id!r} has "
                f"conflicting {column!r} values."
            )


def test_supported_runtime_csvs_have_expected_schema_and_partitions() -> None:
    runtime_cases = [
        ("fungtion_dataset.csv", {"train", "test"}),
        ("effector_finetune_dataset.csv", {"train", "test"}),
        ("effector_pretrain_dataset.csv", {"train"}),
    ]

    for file_name, required_partitions in runtime_cases:
        dataset_path = DATA_ROOT / "csv_dataset" / file_name
        df = load_labeled_dataset(
            dataset_path,
            required_partitions=required_partitions,
        )

        assert {"sequence_id", "label", "partition"}.issubset(df.columns)
        assert required_partitions.issubset(set(df["partition"].unique()))


def test_provenance_membership_audit_rejects_repeated_test_ids() -> None:
    provenance = pd.DataFrame(
        [
            ("duplicate-test", "AAA", 1, "test", 10),
            ("duplicate-test", "AAA", 1, "test", 10),
        ],
        columns=["sequence_id", "sequence", "label", "partition", "cluster_id"],
    )

    with pytest.raises(
        AssertionError,
        match="duplicate-test.*exactly one train and one pretrain",
    ):
        _assert_valid_repeated_provenance_memberships(provenance)


def test_provenance_construction_artifacts_explain_tracked_effector_datasets() -> None:
    combined_positives = _read_provenance_csv(
        DATA_ROOT / "dataset_construction" / "combined_positives.csv"
    )
    filtered_negatives = pd.read_csv(
        (
            DATA_ROOT
            / "dataset_construction"
            / "filtered_new_negative_representatives.csv"
        ),
        dtype={"sequence_id": str},
    )
    effector_dataset = _read_provenance_csv(
        DATA_ROOT / "csv_dataset" / "effector_dataset.csv"
    )
    effector_pretrain = load_labeled_dataset(
        DATA_ROOT / "csv_dataset" / "effector_pretrain_dataset.csv",
        required_partitions={"train"},
    )
    effector_finetune = load_labeled_dataset(
        DATA_ROOT / "csv_dataset" / "effector_finetune_dataset.csv",
        required_partitions={"train", "test"},
    )

    effector_positive = effector_dataset[effector_dataset["label"] == 1].copy()
    effector_negative = effector_dataset[effector_dataset["label"] == 0].copy()

    assert_frame_equal(
        _normalize_frame(combined_positives),
        _normalize_frame(effector_positive),
        check_dtype=False,
    )
    assert set(effector_negative["sequence_id"]) == set(
        filtered_negatives["sequence_id"]
    )
    assert len(effector_negative) == len(filtered_negatives)

    _assert_valid_repeated_provenance_memberships(effector_dataset)
    expected_pretrain = effector_dataset[
        effector_dataset["partition"].isin({"train", "pretrain"})
    ].copy()
    expected_pretrain = expected_pretrain.drop_duplicates(
        subset=["sequence_id"], keep="first"
    )
    expected_pretrain["partition"] = "train"
    assert expected_pretrain["sequence_id"].is_unique
    assert set(
        effector_finetune.loc[effector_finetune["partition"] == "train", "sequence_id"]
    ).issubset(set(effector_pretrain["sequence_id"]))
    assert set(effector_pretrain["sequence_id"]).isdisjoint(
        set(
            effector_finetune.loc[
                effector_finetune["partition"] == "test", "sequence_id"
            ]
        )
    )
    expected_pretrain = _normalize_frame(expected_pretrain)
    assert_frame_equal(
        expected_pretrain,
        _normalize_frame(effector_pretrain),
        check_dtype=False,
    )

    is_positive = effector_dataset["label"] == 1
    is_negative = effector_dataset["label"] == 0
    is_train = effector_dataset["partition"] == "train"
    is_test = effector_dataset["partition"] == "test"
    expected_finetune = pd.concat(
        [
            effector_dataset[is_positive & is_train],
            effector_dataset[
                is_negative & effector_dataset["partition"].isin(["train", "test"])
            ],
            effector_dataset[is_test],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["sequence_id"])

    assert_frame_equal(
        _normalize_frame(expected_finetune),
        _normalize_frame(effector_finetune),
        check_dtype=False,
    )
