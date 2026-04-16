from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from effector_bincls.data import load_labeled_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "src" / "data"


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["sequence_id", "partition"]).reset_index(drop=True)


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


def test_provenance_construction_artifacts_explain_tracked_effector_datasets() -> None:
    combined_positives = load_labeled_dataset(
        DATA_ROOT / "dataset_construction" / "combined_positives.csv",
        required_partitions={"train", "test", "pretrain"},
    )
    filtered_negatives = pd.read_csv(
        (
            DATA_ROOT
            / "dataset_construction"
            / "filtered_new_negative_representatives.csv"
        ),
        dtype={"sequence_id": str},
    )
    effector_dataset = load_labeled_dataset(
        DATA_ROOT / "csv_dataset" / "effector_dataset.csv",
        required_partitions={"train", "test", "pretrain"},
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

    expected_pretrain = effector_dataset[
        effector_dataset["partition"].isin({"train", "pretrain"})
    ].copy()
    expected_pretrain["partition"] = "train"
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
