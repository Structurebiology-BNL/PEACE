"""Package entrypoint for baseline test evaluation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from effector_bincls.checkpoints import load_baseline_model
from effector_bincls.evaluation.common import (
    add_high_recall_summary,
    collect_oof_predictions,
    find_global_optimal_threshold,
    parse_evaluation_args,
    pool_fold_predictions,
)
from effector_bincls.evaluation.reporting import (
    plot_test_metrics,
    plot_threshold_analysis,
)
from effector_bincls.metrics import multi_scores
from effector_bincls.run_utils import (
    convert_to_serializable,
    load_run_config,
    resolve_device,
    setup_logger,
)
from effector_bincls.training.data import load_test_data


def get_baseline_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> np.ndarray:
    """Run baseline checkpoint inference over a labeled data loader."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for features, _ in data_loader:
            if isinstance(features, tuple):
                features = tuple(feature.to(device) for feature in features)
                outputs = model(*features)
            else:
                features = features.to(device)
                outputs = model(features)

            if torch.is_tensor(outputs):
                logits = outputs
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")

            if logits.dim() == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds)


def evaluate_baseline_on_test(
    config: ConfigDict,
    run_dir: Path,
    test_loader,
    test_labels: np.ndarray,
    optimal_threshold: float,
    threshold_method: str,
    device: torch.device,
    logger: logging.Logger,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate baseline checkpoints on the test set using the pooled OOF threshold."""
    logger.info("Evaluating baseline model on test set...")

    num_folds = config.training.num_folds
    all_test_predictions = []

    for fold in range(1, num_folds + 1):
        model_path = run_dir / f"fold_{fold}/checkpoint.pt"

        if model_path.exists():
            model = load_baseline_model(model_path, config, device)
            fold_preds = get_baseline_predictions(model, test_loader, device)
            all_test_predictions.append(fold_preds)
            logger.info("  Baseline predictions for fold %s", fold)
        else:
            logger.warning("  Model not found for fold %s: %s", fold, model_path)

    if not all_test_predictions:
        raise ValueError("No test predictions collected from any fold")

    test_predictions = np.stack(all_test_predictions)
    logger.info("Test predictions stack shape: %s", test_predictions.shape)

    if test_predictions.ndim == 3:
        test_predictions = test_predictions.squeeze(axis=2)
    logger.info("Test predictions after squeeze: %s", test_predictions.shape)

    ensemble_test_preds = np.mean(test_predictions.T, axis=1)
    logger.info("Ensemble test predictions shape: %s", ensemble_test_preds.shape)

    test_labels = test_labels.astype(np.int64)
    ensemble_test_preds = ensemble_test_preds.astype(np.float64)

    if ensemble_test_preds.ndim > 1:
        ensemble_test_preds = ensemble_test_preds.ravel()
    if test_labels.ndim > 1:
        test_labels = test_labels.ravel()

    logger.info(
        "Test predictions shape: %s, dtype: %s",
        ensemble_test_preds.shape,
        ensemble_test_preds.dtype,
    )
    logger.info(
        "Test labels shape: %s, dtype: %s",
        test_labels.shape,
        test_labels.dtype,
    )
    logger.info("Test unique labels: %s", np.unique(test_labels))
    logger.info(
        "Test prediction range: [%.4f, %.4f]",
        ensemble_test_preds.min(),
        ensemble_test_preds.max(),
    )

    try:
        logger.info("Generating threshold analysis plot for test predictions...")
        plot_threshold_analysis(
            outputs=ensemble_test_preds,
            labels=test_labels,
            save_dir=output_dir,
            fold_number="test",
            optimal_threshold=optimal_threshold,
            threshold_method_used=threshold_method,
            logger=logger,
        )
    except Exception as exc:
        logger.warning("Could not create threshold analysis plot: %s", exc)

    test_metrics = multi_scores(
        test_labels, ensemble_test_preds, threshold=optimal_threshold
    )

    logger.info("\nTest Set Results - Baseline Model:")
    logger.info("=" * 60)
    logger.info("Threshold-Independent Metrics:")
    for metric_name in [
        "roc_auc",
        "auprc",
        "high_recall_auprc_0.7",
        "high_recall_auprc_0.8",
    ]:
        if metric_name in test_metrics:
            logger.info("  %-20s: %.4f", metric_name, test_metrics[metric_name])

    logger.info(
        "\nThreshold-Dependent Metrics (threshold=%.4f):",
        optimal_threshold,
    )
    for metric_name in ["accuracy", "f1", "mcc", "precision", "recall"]:
        if metric_name in test_metrics:
            logger.info("  %-20s: %.4f", metric_name, test_metrics[metric_name])

    return {
        "test_metrics": test_metrics,
        "optimal_threshold": optimal_threshold,
        "test_set_stats": {
            "total_samples": len(test_labels),
            "positive_samples": int(np.sum(test_labels)),
            "negative_samples": int(len(test_labels) - np.sum(test_labels)),
        },
    }


def main() -> None:
    """Run baseline test evaluation from a saved training run directory."""
    args = parse_evaluation_args("Test evaluation for baseline model")

    config_path = args.run_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_run_config(args.run_dir)
    logger = setup_logger(
        output_dir=args.run_dir,
        name="baseline_evaluate_test",
        log_file_name="test",
    )

    try:
        device = resolve_device(config)
        logger.info("Using device: %s", device)

        if not args.test_csv.exists():
            raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
        logger.info("Using test CSV file: %s", args.test_csv)

        logger.info("Loading test data...")
        test_loader = load_test_data(config, logger=logger, test_csv_path=args.test_csv)
        test_labels = np.concatenate(
            [label.cpu().numpy() for _, label in test_loader]
        ).ravel()
        logger.info("Test dataset size: %s samples", len(test_loader.dataset))

        fold_predictions, fold_labels = collect_oof_predictions(
            args.run_dir,
            logger,
            prediction_key="predictions",
        )

        optimal_threshold = find_global_optimal_threshold(
            fold_predictions,
            fold_labels,
            args.threshold_method,
            args.target_recall,
            logger,
        )

        try:
            logger.info("Generating threshold analysis plot for OOF predictions...")
            pooled_oof_predictions, pooled_oof_labels = pool_fold_predictions(
                fold_predictions,
                fold_labels,
            )

            plot_threshold_analysis(
                outputs=pooled_oof_predictions,
                labels=pooled_oof_labels,
                save_dir=args.run_dir,
                fold_number="oof",
                optimal_threshold=optimal_threshold,
                threshold_method_used=args.threshold_method,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Could not create OOF threshold analysis plot: %s", exc)

        results = evaluate_baseline_on_test(
            config=config,
            run_dir=args.run_dir,
            test_loader=test_loader,
            test_labels=test_labels,
            optimal_threshold=optimal_threshold,
            threshold_method=args.threshold_method,
            device=device,
            logger=logger,
            output_dir=args.run_dir,
        )

        eval_results_file = args.run_dir / "test_evaluation.yaml"
        results["ensemble_method"] = "simple_average"
        results["threshold_method"] = args.threshold_method
        add_high_recall_summary(results)

        with eval_results_file.open("w") as handle:
            yaml.safe_dump(
                convert_to_serializable(results),
                handle,
                indent=4,
                sort_keys=False,
                default_flow_style=False,
            )

        logger.info("\nTest evaluation results saved to: %s", eval_results_file)

        try:
            plot_test_metrics(results, args.run_dir)
            logger.info("Test metrics plot saved in: %s", args.run_dir)
        except Exception as exc:
            logger.warning("Could not create test metrics plot: %s", exc)

        logger.info("Baseline test evaluation completed successfully!")
        logger.info("Ensemble method: simple_average")
    except Exception as exc:
        logger.error("Baseline test evaluation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
