"""Dataset contract helpers for package-owned runtime workflows."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd

DEFAULT_SEQUENCE_ID_COLUMN = "sequence_id"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_PARTITION_COLUMN = "partition"


def resolve_label_columns(
    label_config: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve sequence-id and label column names from the config contract."""
    label_config = label_config or {}
    sequence_id_column = label_config.get(
        "sequence_id_column",
        DEFAULT_SEQUENCE_ID_COLUMN,
    )
    label_column = label_config.get("label_column", DEFAULT_LABEL_COLUMN)
    return sequence_id_column, label_column


def load_labeled_dataset(
    csv_path: str | Path,
    *,
    label_config: Mapping[str, str] | None = None,
    required_partitions: Collection[str] | None = None,
) -> pd.DataFrame:
    """Load a labeled runtime dataset and validate the supported CSV contract."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    sequence_id_column, label_column = resolve_label_columns(label_config)
    df = pd.read_csv(path, dtype={sequence_id_column: str})

    required_columns = [
        sequence_id_column,
        label_column,
        DEFAULT_PARTITION_COLUMN,
    ]
    missing_columns = sorted(
        column for column in required_columns if column not in df.columns
    )
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path}: {missing_columns}. "
            f"Expected columns: {required_columns}."
        )

    if required_partitions:
        available_partitions = set(df[DEFAULT_PARTITION_COLUMN].dropna().astype(str))
        missing_partitions = sorted(set(required_partitions) - available_partitions)
        if missing_partitions:
            raise ValueError(
                f"Missing required partitions in {path}: {missing_partitions}. "
                f"Available partitions: {sorted(available_partitions)}."
            )

    return df


def validate_two_stage_dataset_pair(
    pretraining_df: pd.DataFrame,
    finetuning_df: pd.DataFrame,
    *,
    pretraining_csv_path: str | Path,
    finetuning_csv_path: str | Path,
    label_config: Mapping[str, str] | None = None,
) -> None:
    """Validate pretraining/finetuning dataset alignment for two-stage workflows."""
    sequence_id_column, label_column = resolve_label_columns(label_config)

    pretraining_labels = dict(
        zip(
            pretraining_df[sequence_id_column],
            pretraining_df[label_column],
            strict=False,
        )
    )
    finetuning_labels = dict(
        zip(
            finetuning_df[sequence_id_column],
            finetuning_df[label_column],
            strict=False,
        )
    )

    pretraining_ids = set(pretraining_labels)
    finetuning_ids = set(finetuning_labels)
    missing_ids = sorted(finetuning_ids - pretraining_ids)
    if missing_ids:
        sample = missing_ids[:5]
        raise ValueError(
            "Finetuning dataset contains samples not in pretraining dataset: "
            f"{sample} from {Path(finetuning_csv_path)}."
        )

    mismatched_ids = sorted(
        sequence_id
        for sequence_id, finetuning_label in finetuning_labels.items()
        if pretraining_labels[sequence_id] != finetuning_label
    )
    if mismatched_ids:
        sample = mismatched_ids[:5]
        raise ValueError(
            "Label inconsistencies detected between datasets: "
            f"{sample} across {Path(pretraining_csv_path)} and "
            f"{Path(finetuning_csv_path)}."
        )
