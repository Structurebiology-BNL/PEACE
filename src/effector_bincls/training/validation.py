"""Validation helpers for retained training workflows."""

from pathlib import Path

from ml_collections import ConfigDict

from effector_bincls.data import load_labeled_dataset, validate_two_stage_dataset_pair


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
