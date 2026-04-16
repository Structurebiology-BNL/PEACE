"""Package-owned data loading and fold orchestration helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import torch
from ml_collections import ConfigDict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from effector_bincls.data import (
    DEFAULT_PARTITION_COLUMN,
    SimpleDataset,
    load_labeled_dataset,
    resolve_label_columns,
    validate_two_stage_dataset_pair,
)


def load_test_data(
    config: ConfigDict,
    logger: logging.Logger | None = None,
    test_csv_path: Optional[Path] = None,
) -> DataLoader:
    """Load test data for package-native evaluation and analysis."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if test_csv_path is not None:
        csv_path = Path(test_csv_path)
    else:
        if hasattr(config.data, "csv_path"):
            csv_path_value = config.data.csv_path
        else:
            csv_path_value = config.data.finetuning_csv_path
        csv_path = Path(csv_path_value)
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, _ = resolve_label_columns(label_config)
    df = load_labeled_dataset(
        csv_path,
        label_config=label_config,
        required_partitions={"test"},
    )
    test_df = df[df[DEFAULT_PARTITION_COLUMN] == "test"]
    if test_df.empty:
        raise ValueError("No test data found in CSV")

    model_type = config.model.type.lower()
    if model_type not in {"simple_predictor", "simple"}:
        raise ValueError(
            "Unsupported model type "
            f"'{model_type}' for test loading. Supported values are "
            "'simple_predictor' and 'simple'."
        )

    use_variants = getattr(config.training, "use_variants", False)
    dataset = SimpleDataset(
        embedding_dir=config.data.embedding_dir,
        csv_path=str(csv_path),
        sequence_ids=test_df[sequence_id_column].tolist(),
        normalize=getattr(config.features, "normalize", True),
        pooling_type=getattr(config.features, "pooling_type", "mean"),
        use_variants=use_variants,
        label_config=label_config,
        logger=logger,
    )
    return DataLoader(
        dataset,
        batch_size=getattr(config.training, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(config.hardware, "num_workers", 0),
        pin_memory=True,
    )


def create_variant_collate_fn(
    variant_sampling_config: dict | None = None,
    random_seed: int | None = None,
    original_variant_index: int = 0,
):
    """Create a collate function with deterministic variant sampling."""
    if random_seed is not None:
        random.seed(random_seed)

    def _variant_collate_fn(batch):
        if len(batch[0]) != 2:
            raise ValueError(
                f"Unexpected batch format with {len(batch[0])} elements per item"
            )

        embeddings_list, labels_list = zip(*batch, strict=False)

        if variant_sampling_config and variant_sampling_config.get("enabled", False):
            embeddings_list = [
                _sample_variants(
                    embeddings,
                    variant_sampling_config.get("num_variants", 4),
                    variant_sampling_config.get("always_include_original", True),
                    original_variant_index=original_variant_index,
                )
                for embeddings in embeddings_list
            ]

        embeddings = torch.stack(list(embeddings_list))
        labels = torch.stack(list(labels_list))
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        return embeddings, labels

    return _variant_collate_fn


def _sample_variants(
    embeddings: torch.Tensor,
    num_variants: int = 4,
    always_include_original: bool = True,
    original_variant_index: int = 0,
) -> torch.Tensor:
    """Sample a fixed number of variants without replacement."""
    if embeddings.dim() < 2:
        return embeddings.unsqueeze(0).repeat(num_variants, *([1] * embeddings.dim()))

    num_available_variants = embeddings.shape[0]
    if num_available_variants < num_variants:
        raise ValueError(
            f"Cannot sample {num_variants} variants: only "
            f"{num_available_variants} available."
        )

    if not 0 <= original_variant_index < num_available_variants:
        raise ValueError(
            "original_variant_index must be within the available variant range"
        )

    if always_include_original and num_available_variants > 0:
        selected_indices = [original_variant_index]
        remaining_to_sample = num_variants - 1
        if remaining_to_sample > 0 and num_available_variants > 1:
            selected_indices.extend(
                random.sample(
                    [
                        index
                        for index in range(num_available_variants)
                        if index != original_variant_index
                    ],
                    remaining_to_sample,
                )
            )
    else:
        selected_indices = random.sample(range(num_available_variants), num_variants)

    return embeddings[torch.tensor(selected_indices[:num_variants])]


def validate_fold_consistency(
    fold_number: int,
    train_ids: list[str],
    val_ids: list[str],
    stage: str,
    previous_folds: list[tuple[int, list[str], list[str]]],
    logger: logging.Logger,
) -> None:
    """Guard against within-fold leakage and cross-fold validation overlap."""
    train_set = set(train_ids)
    val_set = set(val_ids)

    train_val_overlap = train_set & val_set
    if train_val_overlap:
        raise ValueError(
            f"{stage} fold {fold_number}: Train/val overlap detected: "
            f"{len(train_val_overlap)} samples"
        )

    for previous_fold, _, previous_val_ids in previous_folds:
        if previous_fold == fold_number:
            continue
        overlap = val_set & set(previous_val_ids)
        if overlap:
            raise ValueError(
                f"{stage} fold {fold_number}: Val contamination with fold "
                f"{previous_fold}: {len(overlap)} samples"
            )

    logger.info("%s fold %s validation: ok", stage, fold_number)


def validate_two_stage_consistency(
    fold_number: int,
    pretrain_train_ids: list[str],
    pretrain_val_ids: list[str],
    finetune_train_ids: list[str],
    finetune_val_ids: list[str],
    pretraining_samples: dict[str, int],
    finetuning_samples: dict[str, int],
    logger: logging.Logger,
) -> None:
    """Ensure finetuning split membership stays aligned with pretraining."""
    if not set(finetune_train_ids).issubset(pretrain_train_ids):
        raise ValueError(
            f"Fold {fold_number}: Finetuning train contains samples not in "
            "pretraining train"
        )
    if not set(finetune_val_ids).issubset(pretrain_val_ids):
        raise ValueError(
            f"Fold {fold_number}: Finetuning val contains samples not in "
            "pretraining val"
        )

    pretrain_train_pos = {
        seq_id for seq_id in pretrain_train_ids if pretraining_samples[seq_id] == 1
    }
    pretrain_val_pos = {
        seq_id for seq_id in pretrain_val_ids if pretraining_samples[seq_id] == 1
    }
    finetune_train_pos = {
        seq_id for seq_id in finetune_train_ids if finetuning_samples[seq_id] == 1
    }
    finetune_val_pos = {
        seq_id for seq_id in finetune_val_ids if finetuning_samples[seq_id] == 1
    }

    if not finetune_train_pos.issubset(pretrain_train_pos):
        raise ValueError(
            f"Fold {fold_number}: Finetuning train positive samples are not "
            "a subset of pretraining positives"
        )
    if not finetune_val_pos.issubset(pretrain_val_pos):
        raise ValueError(
            f"Fold {fold_number}: Finetuning val positive samples are not "
            "a subset of pretraining positives"
        )

    logger.info("Fold %s two-stage consistency validation: ok", fold_number)


def create_pretraining_folds(
    pretraining_ids: set[str],
    pretraining_samples: dict[str, int],
    num_folds: int,
    random_seed: int,
    logger: logging.Logger,
) -> dict[int, tuple[list[str], list[str]]]:
    """Create stratified pretraining folds."""
    pretraining_ids_list = list(pretraining_ids)
    labels = [pretraining_samples[seq_id] for seq_id in pretraining_ids_list]
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    fold_data: dict[int, tuple[list[str], list[str]]] = {}
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(pretraining_ids_list, labels),
        start=1,
    ):
        train_ids = [pretraining_ids_list[index] for index in train_idx]
        val_ids = [pretraining_ids_list[index] for index in val_idx]
        fold_data[fold] = (train_ids, val_ids)
        logger.info(
            "Pretraining fold %s: %s train, %s val samples",
            fold,
            len(train_ids),
            len(val_ids),
        )
    return fold_data


def create_finetuning_fold_data(
    pretrain_train_ids: list[str],
    pretrain_val_ids: list[str],
    pretraining_samples: dict[str, int],
    finetuning_ids: set[str],
    neg_to_pos_ratio: float | None,
    random_seed: int,
    fold_number: int,
) -> tuple[list[str], list[str]]:
    """Create finetuning splits constrained to the pretraining fold membership."""
    pretrain_train_pos = {
        seq_id for seq_id in pretrain_train_ids if pretraining_samples[seq_id] == 1
    }
    pretrain_val_pos = {
        seq_id for seq_id in pretrain_val_ids if pretraining_samples[seq_id] == 1
    }
    pretrain_train_neg = {
        seq_id for seq_id in pretrain_train_ids if pretraining_samples[seq_id] == 0
    }
    pretrain_val_neg = {
        seq_id for seq_id in pretrain_val_ids if pretraining_samples[seq_id] == 0
    }

    finetune_train_pos = pretrain_train_pos & finetuning_ids
    finetune_val_pos = pretrain_val_pos & finetuning_ids
    finetune_train_neg = pretrain_train_neg & finetuning_ids
    finetune_val_neg = pretrain_val_neg & finetuning_ids

    if neg_to_pos_ratio is not None:
        target_train_neg_count = int(len(finetune_train_pos) * neg_to_pos_ratio)
        if target_train_neg_count < len(finetune_train_neg):
            random.seed(random_seed + fold_number)
            finetune_train_neg = set(
                random.sample(list(finetune_train_neg), target_train_neg_count)
            )

    return (
        list(finetune_train_pos) + list(finetune_train_neg),
        list(finetune_val_pos) + list(finetune_val_neg),
    )


def create_data_loaders(
    final_train_ids: list[str],
    final_val_ids: list[str],
    csv_path: str | Path,
    config: ConfigDict,
    use_variants: bool,
    variant_sampling_config: dict | None,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    """Create package-native train/val data loaders."""
    embedding_dir = config.data.embedding_dir
    normalize = getattr(config.features, "normalize", True)
    pooling_type = getattr(config.features, "pooling_type", "mean")
    batch_size = getattr(config.training, "batch_size", 32)
    label_config = getattr(config.data, "label_config", {})

    train_dataset = SimpleDataset(
        embedding_dir=embedding_dir,
        csv_path=str(csv_path),
        sequence_ids=final_train_ids,
        normalize=normalize,
        pooling_type=pooling_type,
        use_variants=use_variants,
        label_config=label_config,
        logger=logger,
    )
    val_dataset = SimpleDataset(
        embedding_dir=embedding_dir,
        csv_path=str(csv_path),
        sequence_ids=final_val_ids,
        normalize=normalize,
        pooling_type=pooling_type,
        use_variants=use_variants,
        label_config=label_config,
        logger=logger,
    )

    if use_variants:
        collate_fn = create_variant_collate_fn(
            variant_sampling_config,
            getattr(config.hardware, "random_seed", 42),
            train_dataset.original_variant_index,
        )
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(config.hardware, "num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(config.hardware, "num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_baseline_data_loaders(
    final_train_ids: list[str],
    final_val_ids: list[str],
    csv_path: str | Path,
    config: ConfigDict,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    """Create baseline-specific data loaders."""
    if config.model.type != "simple_predictor":
        raise ValueError(f"Unsupported baseline model type: {config.model.type}")
    return create_data_loaders(
        final_train_ids,
        final_val_ids,
        csv_path,
        config,
        use_variants=False,
        variant_sampling_config=None,
        logger=logger,
    )


def create_baseline_data_loader_fn(
    config: ConfigDict,
    logger: logging.Logger | None = None,
):
    """Create a fold loader factory for baseline training."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if not hasattr(config.data, "csv_path"):
        raise ValueError("Configuration must specify data.csv_path")

    csv_path = Path(config.data.csv_path)
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, label_column = resolve_label_columns(label_config)
    df = load_labeled_dataset(
        csv_path,
        label_config=label_config,
        required_partitions={"train", "test"},
    )
    train_df = df[df[DEFAULT_PARTITION_COLUMN] == "train"].copy()
    if train_df.empty:
        raise ValueError(f"No training data found in CSV: {csv_path}")

    sequence_ids = train_df[sequence_id_column].tolist()
    labels = train_df[label_column].tolist()
    num_folds = getattr(config.training, "num_folds", 5)
    random_seed = getattr(config.hardware, "random_seed", 42)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    fold_data = {
        fold: (
            [sequence_ids[index] for index in train_idx],
            [sequence_ids[index] for index in val_idx],
        )
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(sequence_ids, labels),
            start=1,
        )
    }

    def data_loader_fn(fold_number: int) -> tuple[DataLoader, DataLoader]:
        if fold_number not in fold_data:
            raise ValueError(f"Fold number must be between 1 and {num_folds}")
        train_ids, val_ids = fold_data[fold_number]
        return create_baseline_data_loaders(
            train_ids,
            val_ids,
            csv_path,
            config,
            logger,
        )

    return data_loader_fn


def create_single_stage_data_loader_fn(
    config: ConfigDict,
    logger: logging.Logger | None = None,
):
    """Create a fold loader factory for single-stage prototype training."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if not hasattr(config.data, "csv_path"):
        raise ValueError("Configuration must specify data.csv_path")

    csv_path = Path(config.data.csv_path)
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, label_column = resolve_label_columns(label_config)
    df = load_labeled_dataset(
        csv_path,
        label_config=label_config,
        required_partitions={"train", "test"},
    )
    train_df = df[df[DEFAULT_PARTITION_COLUMN] == "train"].copy()
    if train_df.empty:
        raise ValueError(f"No training data found in CSV: {csv_path}")

    sequence_ids = train_df[sequence_id_column].tolist()
    labels = train_df[label_column].tolist()
    num_folds = getattr(config.training, "num_folds", 5)
    random_seed = getattr(config.hardware, "random_seed", 42)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    fold_data = {
        fold: (
            [sequence_ids[index] for index in train_idx],
            [sequence_ids[index] for index in val_idx],
        )
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(sequence_ids, labels),
            start=1,
        )
    }

    def data_loader_fn(fold_number: int) -> tuple[DataLoader, DataLoader]:
        if fold_number not in fold_data:
            raise ValueError(f"Fold number must be between 1 and {num_folds}")
        train_ids, val_ids = fold_data[fold_number]
        variant_sampling_config = getattr(config.training, "variant_sampling", {})
        use_variants = variant_sampling_config.get("enabled", False)
        if not use_variants:
            variant_sampling_config = None
        return create_data_loaders(
            train_ids,
            val_ids,
            csv_path,
            config,
            use_variants,
            variant_sampling_config,
            logger,
        )

    return data_loader_fn


def create_two_stage_data_loader_fn(
    config: ConfigDict,
    logger: logging.Logger | None = None,
):
    """Create a package-owned two-stage data loader with explicit consistency checks."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if not hasattr(config.data, "pretraining_csv_path"):
        raise ValueError("Configuration must specify data.pretraining_csv_path")
    if not hasattr(config.data, "finetuning_csv_path"):
        raise ValueError("Configuration must specify data.finetuning_csv_path")

    pretraining_csv_path = Path(config.data.pretraining_csv_path)
    finetuning_csv_path = Path(config.data.finetuning_csv_path)
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, label_column = resolve_label_columns(label_config)
    pretraining_df = load_labeled_dataset(
        pretraining_csv_path,
        label_config=label_config,
        required_partitions={"train"},
    )
    finetuning_df = load_labeled_dataset(
        finetuning_csv_path,
        label_config=label_config,
        required_partitions={"train", "test"},
    )
    validate_two_stage_dataset_pair(
        pretraining_df,
        finetuning_df,
        pretraining_csv_path=pretraining_csv_path,
        finetuning_csv_path=finetuning_csv_path,
        label_config=label_config,
    )

    pretraining_train_df = pretraining_df[
        pretraining_df[DEFAULT_PARTITION_COLUMN] == "train"
    ].copy()
    finetuning_train_df = finetuning_df[
        finetuning_df[DEFAULT_PARTITION_COLUMN] == "train"
    ].copy()
    if pretraining_train_df.empty:
        raise ValueError(
            f"No training data found in pretraining CSV: {pretraining_csv_path}"
        )
    if finetuning_train_df.empty:
        raise ValueError(
            f"No training data found in finetuning CSV: {finetuning_csv_path}"
        )

    pretraining_samples = dict(
        zip(
            pretraining_train_df[sequence_id_column],
            pretraining_train_df[label_column],
            strict=False,
        )
    )
    finetuning_samples = dict(
        zip(
            finetuning_train_df[sequence_id_column],
            finetuning_train_df[label_column],
            strict=False,
        )
    )
    pretraining_ids = set(pretraining_samples)
    finetuning_ids = set(finetuning_samples)

    num_folds = getattr(config.training, "num_folds", 5)
    random_seed = getattr(config.hardware, "random_seed", 42)
    pretraining_fold_data = create_pretraining_folds(
        pretraining_ids,
        pretraining_samples,
        num_folds,
        random_seed,
        logger,
    )
    previous_folds: list[tuple[int, list[str], list[str]]] = []
    stored_pretraining_folds: dict[int, tuple[list[str], list[str]]] = {}

    def data_loader_fn(
        fold_number: int,
        phase: str | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        if fold_number not in pretraining_fold_data:
            raise ValueError(f"Fold number must be between 1 and {num_folds}")

        pretrain_train_ids, pretrain_val_ids = pretraining_fold_data[fold_number]
        if phase == "pretraining":
            csv_path = pretraining_csv_path
            variant_sampling_config = getattr(config.training, "variant_sampling", {})
            use_variants = variant_sampling_config.get("enabled", False)
            if not use_variants:
                variant_sampling_config = None
            final_train_ids = pretrain_train_ids
            final_val_ids = pretrain_val_ids
            stored_pretraining_folds[fold_number] = (final_train_ids, final_val_ids)
        elif phase == "finetuning":
            csv_path = finetuning_csv_path
            variant_sampling_config = getattr(config.training, "variant_sampling", {})
            use_variants = variant_sampling_config.get("enabled", False)
            if not use_variants:
                variant_sampling_config = None
            final_train_ids, final_val_ids = create_finetuning_fold_data(
                pretrain_train_ids,
                pretrain_val_ids,
                pretraining_samples,
                finetuning_ids,
                getattr(config.training.finetuning, "neg_to_pos_ratio", None),
                random_seed,
                fold_number,
            )
            validate_two_stage_consistency(
                fold_number,
                stored_pretraining_folds[fold_number][0],
                stored_pretraining_folds[fold_number][1],
                final_train_ids,
                final_val_ids,
                pretraining_samples,
                finetuning_samples,
                logger,
            )
        else:
            raise ValueError("phase must be 'pretraining' or 'finetuning'")

        validate_fold_consistency(
            fold_number,
            final_train_ids,
            final_val_ids,
            phase.capitalize(),
            previous_folds,
            logger,
        )
        previous_folds.append((fold_number, final_train_ids, final_val_ids))
        return create_data_loaders(
            final_train_ids,
            final_val_ids,
            csv_path,
            config,
            use_variants,
            variant_sampling_config,
            logger,
        )

    return data_loader_fn
