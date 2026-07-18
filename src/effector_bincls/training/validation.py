"""Validation helpers for retained training workflows."""

import math
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict

from effector_bincls.data import (
    DEFAULT_PARTITION_COLUMN,
    load_labeled_dataset,
    open_packed_embedding_dataset,
    require_sequence_indices,
    resolve_label_columns,
    validate_two_stage_dataset_pair,
)


def _require_section(container: Any, name: str, context: str) -> Any:
    section = getattr(container, name, None)
    if section is None:
        raise ValueError(f"{context} requires a {name} section.")
    return section


def _require_non_empty_string(
    section: Any,
    name: str,
    qualified_name: str,
) -> str:
    value = getattr(section, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{qualified_name} must be non-empty.")
    return value


def _require_bool(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    expected: bool | None = None,
) -> bool:
    value = getattr(section, name, None)
    if not isinstance(value, bool):
        raise ValueError(f"{qualified_name} must be boolean, got {value!r}.")
    if expected is not None and value is not expected:
        expected_text = str(expected).lower()
        raise ValueError(f"Contrastive-BCE requires {qualified_name}={expected_text}.")
    return value


def _integer_requirement(qualified_name: str, minimum: int) -> str:
    if minimum == 1:
        return f"{qualified_name} must be a positive integer"
    if minimum == 2:
        return f"{qualified_name} must be at least 2"
    return f"{qualified_name} must be an integer >= {minimum}"


def _require_integer(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    minimum: int,
) -> int:
    value = getattr(section, name, None)
    requirement = _integer_requirement(qualified_name, minimum)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{requirement}, got {value!r}.")
    return value


def _require_exact_integer(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    expected: int,
) -> int:
    value = getattr(section, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"Contrastive-BCE requires {qualified_name}={expected}.")
    return value


def _finite_requirement(
    qualified_name: str,
    minimum: float | None,
    maximum: float | None,
    minimum_inclusive: bool,
    maximum_inclusive: bool,
) -> str:
    if minimum == 0.0 and maximum is None:
        operator = ">=" if minimum_inclusive else ">"
        return f"{qualified_name} must be finite and {operator} 0"
    if minimum == 0.0 and maximum == 1.0:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        return f"{qualified_name} must be in {left}0, 1{right}"
    return f"{qualified_name} is outside its supported finite range"


def _require_finite(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    value = getattr(section, name, None)
    requirement = _finite_requirement(
        qualified_name,
        minimum,
        maximum,
        minimum_inclusive,
        maximum_inclusive,
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{requirement}, got {value!r}.")
    try:
        numeric_value = float(value)
    except OverflowError:
        raise ValueError(f"{requirement}, got {value!r}.") from None
    invalid = not math.isfinite(numeric_value)
    if minimum is not None:
        invalid = invalid or (
            numeric_value < minimum if minimum_inclusive else numeric_value <= minimum
        )
    if maximum is not None:
        invalid = invalid or (
            numeric_value > maximum if maximum_inclusive else numeric_value >= maximum
        )
    if invalid:
        raise ValueError(f"{requirement}, got {value!r}.")
    return numeric_value


def _require_choice(
    section: Any,
    name: str,
    qualified_name: str,
    choices: set[str],
) -> str:
    value = getattr(section, name, None)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{qualified_name} must be one of {sorted(choices)}, got {value!r}."
        )
    return value


def _validate_plateau_scheduler(scheduler: Any) -> None:
    if hasattr(scheduler, "factor"):
        _require_finite(
            scheduler,
            "factor",
            "training.lr_scheduler.factor",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
    for name in ("patience", "cooldown"):
        if hasattr(scheduler, name):
            _require_integer(
                scheduler,
                name,
                f"training.lr_scheduler.{name}",
                minimum=0,
            )
    for name in ("threshold", "min_lr"):
        if hasattr(scheduler, name):
            _require_finite(
                scheduler,
                name,
                f"training.lr_scheduler.{name}",
                minimum=0.0,
            )
    if hasattr(scheduler, "threshold_mode"):
        _require_choice(
            scheduler,
            "threshold_mode",
            "training.lr_scheduler.threshold_mode",
            {"rel", "abs"},
        )


def _require_model_type(
    config: ConfigDict,
    *,
    expected: str,
    entrypoint_name: str,
) -> None:
    model_type = getattr(getattr(config, "model", None), "type", None)
    if model_type != expected:
        raise ValueError(
            f"{entrypoint_name} requires model.type='{expected}', got '{model_type}'."
        )


def _require_absent_training_sections(
    config: ConfigDict,
    *,
    entrypoint_name: str,
) -> None:
    training_config = getattr(config, "training", None)
    if training_config is None:
        raise ValueError(f"{entrypoint_name} requires a training section.")

    has_staged_training = hasattr(training_config, "pretraining") or hasattr(
        training_config, "finetuning"
    )
    if has_staged_training:
        raise ValueError(
            f"{entrypoint_name} requires single-stage configuration. "
            "Use the two-stage entrypoint for configs with training.pretraining and "
            "training.finetuning sections."
        )


def validate_baseline_training_config(config: ConfigDict) -> None:
    """Validate baseline training configuration."""
    _require_model_type(
        config,
        expected="simple_predictor",
        entrypoint_name="Baseline training",
    )
    _require_absent_training_sections(
        config,
        entrypoint_name="Baseline training",
    )


def validate_contrastive_bce_config(config: ConfigDict) -> None:
    """Validate the complete public Contrastive-BCE configuration contract."""
    data = _require_section(config, "data", "Contrastive-BCE")
    features = _require_section(config, "features", "Contrastive-BCE")
    model = _require_section(config, "model", "Contrastive-BCE")
    training = _require_section(config, "training", "Contrastive-BCE")
    output = _require_section(config, "output", "Contrastive-BCE")
    hardware = _require_section(config, "hardware", "Contrastive-BCE")

    validate_baseline_training_config(config)
    for name in ("csv_path", "embedding_dir", "results_dir"):
        _require_non_empty_string(data, name, f"data.{name}")
    _require_bool(features, "normalize", "features.normalize")
    _require_non_empty_string(features, "pooling_type", "features.pooling_type")

    _require_exact_integer(model, "output_dim", "model.output_dim", expected=1)
    _require_bool(
        model,
        "use_contrastive",
        "model.use_contrastive",
        expected=True,
    )
    for name in ("input_dim", "encoder_hidden_dim", "contrastive_dim"):
        _require_integer(model, name, f"model.{name}", minimum=1)
    _require_finite(
        model,
        "dropout_rate",
        "model.dropout_rate",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )

    _require_integer(training, "batch_size", "training.batch_size", minimum=1)
    _require_integer(training, "num_folds", "training.num_folds", minimum=2)
    num_epochs = _require_integer(
        training,
        "num_epochs",
        "training.num_epochs",
        minimum=1,
    )
    _require_finite(
        training,
        "learning_rate",
        "training.learning_rate",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_finite(
        training,
        "weight_decay",
        "training.weight_decay",
        minimum=0.0,
    )
    warmup_epochs = _require_integer(
        training,
        "warmup_epochs",
        "training.warmup_epochs",
        minimum=0,
    )
    if warmup_epochs > num_epochs:
        raise ValueError("training.warmup_epochs must not exceed training.num_epochs.")
    _require_integer(
        training,
        "early_stopping_patience",
        "training.early_stopping_patience",
        minimum=1,
    )
    _require_finite(
        training,
        "grad_clip_value",
        "training.grad_clip_value",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_choice(
        training,
        "threshold_method",
        "training.threshold_method",
        {"youden", "f1", "mcc", "recall_constrained"},
    )
    _require_finite(
        training,
        "target_recall",
        "training.target_recall",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _require_choice(
        training,
        "monitor_metric",
        "training.monitor_metric",
        {"loss", "auprc", "roc_auc"},
    )
    mode = _require_choice(
        training,
        "mode",
        "training.mode",
        {"min", "max"},
    )
    expected_mode = "min" if training.monitor_metric == "loss" else "max"
    if mode != expected_mode:
        raise ValueError(
            f"Contrastive-BCE requires training.mode='{expected_mode}' for "
            f"monitor_metric='{training.monitor_metric}'."
        )

    if getattr(training, "loss_type", None) != "contrastive_bce":
        raise ValueError(
            "Contrastive-BCE requires training.loss_type='contrastive_bce'."
        )
    for name in ("bce_weight", "unsupervised_weight", "temperature"):
        _require_finite(
            training,
            name,
            f"training.{name}",
            minimum=0.0,
            minimum_inclusive=False,
        )
    _require_bool(
        training,
        "use_variants",
        "training.use_variants",
        expected=True,
    )
    variant_sampling = _require_section(
        training,
        "variant_sampling",
        "Contrastive-BCE training",
    )
    _require_bool(
        variant_sampling,
        "enabled",
        "training.variant_sampling.enabled",
        expected=True,
    )
    _require_integer(
        variant_sampling,
        "num_variants",
        "training.variant_sampling.num_variants",
        minimum=2,
    )
    _require_bool(
        variant_sampling,
        "always_include_original",
        "training.variant_sampling.always_include_original",
        expected=True,
    )

    scheduler = _require_section(
        training,
        "lr_scheduler",
        "Contrastive-BCE training",
    )
    scheduler_type = _require_choice(
        scheduler,
        "scheduler_type",
        "training.lr_scheduler.scheduler_type",
        {"plateau", "cosine"},
    )
    if scheduler_type == "cosine":
        eta_min = _require_finite(
            scheduler,
            "eta_min",
            "training.lr_scheduler.eta_min",
            minimum=0.0,
        )
        if eta_min >= float(training.learning_rate):
            raise ValueError(
                "training.lr_scheduler.eta_min must be less than "
                "training.learning_rate."
            )
    else:
        _validate_plateau_scheduler(scheduler)

    _require_bool(output, "save_checkpoints", "output.save_checkpoints")
    _require_bool(
        output,
        "plot_training_curves",
        "output.plot_training_curves",
    )
    _require_integer(hardware, "gpu_id", "hardware.gpu_id", minimum=-1)
    _require_integer(
        hardware,
        "random_seed",
        "hardware.random_seed",
        minimum=0,
    )
    _require_bool(hardware, "deterministic", "hardware.deterministic")
    _require_bool(hardware, "debug_logging", "hardware.debug_logging")
    _require_integer(
        hardware,
        "num_workers",
        "hardware.num_workers",
        minimum=0,
    )


def validate_contrastive_bce_inputs(config: ConfigDict) -> None:
    """Validate runtime data and packed embeddings without creating run output."""
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, label_column = resolve_label_columns(label_config)
    dataframe = load_labeled_dataset(
        config.data.csv_path,
        label_config=label_config,
        required_partitions={"train", "test"},
    )

    required_rows = dataframe[
        dataframe[DEFAULT_PARTITION_COLUMN].isin({"train", "test"})
    ]
    train_rows = required_rows[required_rows[DEFAULT_PARTITION_COLUMN] == "train"]
    train_labels = set(train_rows[label_column].tolist())
    if train_labels != {0, 1}:
        raise ValueError(
            "Contrastive-BCE train partition must contain both labels 0 and 1; "
            f"got {sorted(train_labels)}."
        )

    class_counts = train_rows[label_column].value_counts()
    num_folds = int(config.training.num_folds)
    underfilled = {
        int(label): int(class_counts.get(label, 0))
        for label in (0, 1)
        if int(class_counts.get(label, 0)) < num_folds
    }
    if underfilled:
        raise ValueError(
            f"Contrastive-BCE requires at least {num_folds} training samples per "
            f"class for {num_folds}-fold stratification; got {underfilled}."
        )

    embeddings, available_ids, metadata = open_packed_embedding_dataset(
        config.data.embedding_dir
    )
    available_variants = int(embeddings.shape[1])
    embedding_dim = int(embeddings.shape[2])
    requested_variants = int(config.training.variant_sampling.num_variants)
    if metadata.get("pooling_type") != config.features.pooling_type:
        raise ValueError(
            "Packed embedding pooling_type "
            f"{metadata.get('pooling_type')!r} does not match configured "
            f"pooling_type {config.features.pooling_type!r}."
        )
    if embedding_dim != int(config.model.input_dim):
        raise ValueError(
            f"Packed embedding_dim {embedding_dim} does not match "
            f"model.input_dim {config.model.input_dim}."
        )
    if available_variants < requested_variants:
        raise ValueError(
            f"Packed embedding dataset contains {available_variants} variants but "
            f"training.variant_sampling requests {requested_variants}."
        )

    original_variant_index = metadata.get("original_variant_index")
    if (
        isinstance(original_variant_index, bool)
        or not isinstance(original_variant_index, int)
        or not 0 <= original_variant_index < available_variants
    ):
        raise ValueError(
            "Packed embedding original_variant_index must be an integer within "
            f"[0, {available_variants}); got {original_variant_index!r}."
        )

    requested_ids = required_rows[sequence_id_column].tolist()
    available_id_set = set(available_ids)
    missing_ids = [
        sequence_id
        for sequence_id in requested_ids
        if sequence_id not in available_id_set
    ]
    if missing_ids:
        raise FileNotFoundError(
            "Packed embedding dataset is missing runtime sequence IDs: "
            f"{missing_ids[:5]} ({len(missing_ids)} total)."
        )
    require_sequence_indices(requested_ids, available_ids)
    del embeddings


def validate_prototype_single_stage_config(config: ConfigDict) -> None:
    """Validate single-stage prototype ranking configuration."""
    training_config = getattr(config, "training", None)
    if training_config is None:
        raise ValueError("Single-stage prototype training requires a training section.")

    contrastive_type = getattr(training_config, "contrastive_type", None)
    if contrastive_type != "prototype_ranking":
        raise ValueError(
            "Single-stage prototype training requires "
            "training.contrastive_type='prototype_ranking', "
            f"got '{contrastive_type}'."
        )

    _require_model_type(
        config,
        expected="simple",
        entrypoint_name="Single-stage prototype training",
    )
    _require_absent_training_sections(
        config,
        entrypoint_name="Single-stage prototype training",
    )


def validate_prototype_two_stage_config(config: ConfigDict) -> None:
    """Validate two-stage prototype ranking configuration."""
    _require_model_type(
        config,
        expected="simple",
        entrypoint_name="Two-stage prototype training",
    )

    training_config = getattr(config, "training", None)
    if training_config is None:
        raise ValueError("Two-stage prototype training requires a training section.")

    required_sections = ("pretraining", "finetuning")
    for section in required_sections:
        if not hasattr(training_config, section):
            raise ValueError(
                f"Two-stage prototype training is missing training.{section}. "
                "Use the single-stage entrypoint for configs without staged sections."
            )

        stage_config = getattr(training_config, section)
        contrastive_type = getattr(stage_config, "contrastive_type", None)
        expected_type = "prototype" if section == "pretraining" else "prototype_ranking"
        if contrastive_type != expected_type:
            raise ValueError(
                "Two-stage prototype training requires "
                f"training.{section}.contrastive_type='{expected_type}', "
                f"got '{contrastive_type}'."
            )

    data_config = getattr(config, "data", None)
    if data_config is None:
        raise ValueError("Two-stage prototype training requires a data section.")

    pretraining_csv = Path(getattr(data_config, "pretraining_csv_path", ""))
    finetuning_csv = Path(getattr(data_config, "finetuning_csv_path", ""))
    label_config = getattr(data_config, "label_config", {})

    if not pretraining_csv.exists():
        raise FileNotFoundError(f"Pretraining dataset not found: {pretraining_csv}")
    if not finetuning_csv.exists():
        raise FileNotFoundError(f"Fine-tuning dataset not found: {finetuning_csv}")

    pretraining_df = load_labeled_dataset(
        pretraining_csv,
        label_config=label_config,
        required_partitions={"train"},
    )
    finetuning_df = load_labeled_dataset(
        finetuning_csv,
        label_config=label_config,
        required_partitions={"train", "test"},
    )
    validate_two_stage_dataset_pair(
        pretraining_df,
        finetuning_df,
        pretraining_csv_path=pretraining_csv,
        finetuning_csv_path=finetuning_csv,
        label_config=label_config,
    )

    pretrained_run_dir = getattr(training_config, "run_dir", None)
    if pretrained_run_dir is not None and not Path(pretrained_run_dir).exists():
        raise FileNotFoundError(
            f"Pretrained model directory not found: {Path(pretrained_run_dir)}"
        )
