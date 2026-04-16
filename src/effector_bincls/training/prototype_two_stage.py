"""Package entrypoint for two-stage prototype contrastive training."""

import argparse
import time
from pathlib import Path

import yaml

from effector_bincls.run_utils import (
    convert_to_serializable,
    log_config_params,
    setup_training,
)
from effector_bincls.training.cross_validation import (
    run_prototype_ranking_two_stage_cv,
)
from effector_bincls.training.validation import validate_prototype_two_stage_config


def main() -> None:
    """Run the two-stage prototype contrastive training pipeline."""
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Two-stage prototype contrastive training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    args = parser.parse_args()

    config, run_dir, logger = setup_training(config_path=args.config)

    try:
        validate_prototype_two_stage_config(config)

        pretraining_csv = config.data.pretraining_csv_path
        finetuning_csv = config.data.finetuning_csv_path
        pretrained_run_dir = getattr(config.training, "run_dir", None)

        if pretrained_run_dir is not None:
            logger.info("=== FINETUNING-ONLY MODE ===")
            logger.info("Loading pretrained models from: %s", pretrained_run_dir)
            logger.info("Stage 1: Skipped (using pretrained models)")
            logger.info("Stage 2: Fine-tuning on %s", Path(finetuning_csv).name)
        else:
            logger.info("=== TWO-STAGE PROTOTYPE CONTRASTIVE TRAINING ===")
            logger.info("Stage 1: Pretraining on %s", Path(pretraining_csv).name)
            logger.info("Stage 2: Fine-tuning on %s", Path(finetuning_csv).name)

        log_config_params(config, logger)
        logger.info("Using simple model for two-stage training")
        logger.info(
            "SimplePredictor embed_dim: %s, contrastive_dim: %s",
            config.model.input_dim,
            getattr(config.model, "contrastive_dim", 128),
        )

        if pretrained_run_dir is None:
            logger.info("Stage 1 - Pretraining:")
            logger.info(
                "  Contrastive type: %s",
                config.training.pretraining.contrastive_type,
            )
            logger.info("  Epochs: %s", config.training.pretraining.num_epochs)
            logger.info(
                "  Learning rate: %s", config.training.pretraining.learning_rate
            )

        logger.info("Stage 2 - Fine-tuning:")
        logger.info(
            "  Contrastive type: %s",
            config.training.finetuning.contrastive_type,
        )
        logger.info("  Epochs: %s", config.training.finetuning.num_epochs)
        logger.info("  Learning rate: %s", config.training.finetuning.learning_rate)

        if pretrained_run_dir is not None:
            logger.info(
                "Starting finetuning-only prototype contrastive cross-validation"
            )
        else:
            logger.info("Starting two-stage prototype contrastive cross-validation")

        results = run_prototype_ranking_two_stage_cv(
            config=config,
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
            )

        logger.info("Results saved to %s", results_file)
        if pretrained_run_dir is not None:
            logger.info(
                "Finetuning-only prototype ranking training completed successfully"
            )
        else:
            logger.info("Two-stage prototype ranking training completed successfully")
        logger.info("Time taken: %.1f seconds", time.time() - start_time)
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
