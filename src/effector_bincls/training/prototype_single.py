"""Package entrypoint for single-stage prototype contrastive training."""

import argparse
import time
import traceback

import yaml

from effector_bincls.run_utils import (
    convert_to_serializable,
    log_config_params,
    setup_training,
)
from effector_bincls.training.cross_validation import run_prototype_ranking_cv
from effector_bincls.training.data import create_single_stage_data_loader_fn
from effector_bincls.training.validation import (
    validate_prototype_single_stage_config,
)


def main() -> None:
    """Run the single-stage prototype contrastive training pipeline."""
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Single-stage prototype contrastive training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    args = parser.parse_args()

    config, run_dir, logger = setup_training(config_path=args.config)

    try:
        validate_prototype_single_stage_config(config)
        log_config_params(config, logger)
        logger.info(
            "Using SimplePredictor model for single-stage "
            "prototype contrastive training"
        )

        data_loader_fn = create_single_stage_data_loader_fn(config, logger)
        logger.info("Starting single-stage prototype contrastive cross-validation")
        results = run_prototype_ranking_cv(
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

        logger.info("Prototype contrastive training completed successfully")
        logger.info("Total training time: %.1f seconds", time.time() - start_time)
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise
    except Exception as exc:
        logger.error("Prototype contrastive training failed with error: %s", exc)
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
