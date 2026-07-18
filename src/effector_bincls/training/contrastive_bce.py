"""Public entrypoint for single-stage Contrastive-BCE training."""

from __future__ import annotations

import argparse
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
from effector_bincls.training.validation import (
    validate_contrastive_bce_config,
    validate_contrastive_bce_inputs,
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
    validate_contrastive_bce_inputs(preflight_config)

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
