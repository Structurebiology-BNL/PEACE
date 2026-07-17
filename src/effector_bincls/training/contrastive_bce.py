"""Public entrypoint for single-stage Contrastive-BCE training."""

from __future__ import annotations

import argparse
import math
import time
import traceback

import yaml
from ml_collections import ConfigDict

from effector_bincls.run_utils import (
    convert_to_serializable,
    load_config,
    log_config_params,
    setup_training,
)
from effector_bincls.training.contrastive_bce_cv import run_contrastive_bce_cv
from effector_bincls.training.data import create_contrastive_bce_data_loader_fn
from effector_bincls.training.validation import validate_baseline_training_config


def _require_positive_integer(section, name: str, qualified_name: str) -> int:
    value = getattr(section, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{qualified_name} must be a positive integer, got {value!r}.")
    return value


def _require_positive_finite(section, name: str, qualified_name: str) -> float:
    value = getattr(section, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{qualified_name} must be finite and > 0, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{qualified_name} must be finite and > 0, got {value!r}.")
    return value


def validate_contrastive_bce_config(config: ConfigDict) -> None:
    """Validate the complete public Contrastive-BCE configuration contract."""
    validate_baseline_training_config(config)
    model = config.model
    training = config.training

    output_dim = getattr(model, "output_dim", None)
    if (
        isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or output_dim != 1
    ):
        raise ValueError("Contrastive-BCE requires model.output_dim=1.")
    if getattr(model, "use_contrastive", None) is not True:
        raise ValueError("Contrastive-BCE requires model.use_contrastive=true.")
    for name in ("input_dim", "encoder_hidden_dim", "contrastive_dim"):
        _require_positive_integer(model, name, f"model.{name}")

    _require_positive_integer(training, "batch_size", "training.batch_size")
    num_folds = _require_positive_integer(
        training,
        "num_folds",
        "training.num_folds",
    )
    if num_folds < 2:
        raise ValueError("training.num_folds must be at least 2.")
    _require_positive_integer(training, "num_epochs", "training.num_epochs")
    if getattr(training, "loss_type", None) != "contrastive_bce":
        raise ValueError(
            "Contrastive-BCE requires training.loss_type='contrastive_bce'."
        )
    for name in ("bce_weight", "unsupervised_weight", "temperature"):
        _require_positive_finite(training, name, f"training.{name}")

    if getattr(training, "use_variants", None) is not True:
        raise ValueError("Contrastive-BCE requires training.use_variants=true.")
    variant_sampling = getattr(training, "variant_sampling", None)
    if variant_sampling is None:
        raise ValueError("Contrastive-BCE requires training.variant_sampling.")
    if getattr(variant_sampling, "enabled", None) is not True:
        raise ValueError(
            "Contrastive-BCE requires training.variant_sampling.enabled=true."
        )
    num_variants = _require_positive_integer(
        variant_sampling,
        "num_variants",
        "training.variant_sampling.num_variants",
    )
    if num_variants < 2:
        raise ValueError("training.variant_sampling.num_variants must be at least 2.")
    if getattr(variant_sampling, "always_include_original", None) is not True:
        raise ValueError(
            "Contrastive-BCE requires "
            "training.variant_sampling.always_include_original=true."
        )


def main() -> None:
    """Run BCE classification regularized by dropout-view InfoNCE."""
    parser = argparse.ArgumentParser(description="Contrastive-BCE training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    args = parser.parse_args()

    preflight_config = ConfigDict(load_config(args.config))
    validate_contrastive_bce_config(preflight_config)

    start_time = time.time()
    config, run_dir, logger = setup_training(config_path=args.config)
    try:
        log_config_params(config, logger)
        data_loader_fn = create_contrastive_bce_data_loader_fn(config, logger)
        results = run_contrastive_bce_cv(
            config=config,
            data_loader_fn=data_loader_fn,
            run_dir=run_dir,
            logger=logger,
        )
        results_file = run_dir / "results.yaml"
        with results_file.open("w") as handle:
            yaml.safe_dump(
                convert_to_serializable(results),
                handle,
                indent=4,
                sort_keys=False,
                default_flow_style=False,
            )
        logger.info("Contrastive-BCE training completed successfully")
        logger.info("Total training time: %.1f seconds", time.time() - start_time)
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise
    except Exception as exc:
        logger.error("Contrastive-BCE training failed with error: %s", exc)
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
