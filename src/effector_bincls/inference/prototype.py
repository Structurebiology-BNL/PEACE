"""Package entrypoint for prototype ranking inference on unseen embeddings."""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from effector_bincls.checkpoints import (
    get_checkpoint_path,
    require_prototype_model_type,
)
from effector_bincls.data import open_packed_embedding_dataset
from effector_bincls.prototype_loading import load_prototype_ranking_model
from effector_bincls.prototype_scoring import compute_prototype_probabilities
from effector_bincls.run_utils import load_run_config, resolve_device, setup_logger


def parse_inference_args() -> argparse.Namespace:
    """Parse command line arguments for prototype inference."""
    parser = argparse.ArgumentParser(
        description="Inference script for prototype ranking model on unseen data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        required=True,
        help="Directory containing a packed embedding dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output CSV file for predictions (default: model_dir/predictions.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for binary predictions",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        default=False,
        help=(
            "Use single-stage training checkpoint paths (fold_X/checkpoint.pt). "
            "Default is two-stage (fold_X/finetuning/checkpoint.pt)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="mean",
        choices=["mean", "max", "bos", "eos"],
        help="Pooling type used in embeddings (must match training)",
    )
    return parser.parse_args()


def load_embedding_batch(
    batch_embeddings: np.ndarray,
    *,
    normalize: bool = True,
    use_variants: bool = False,
    original_variant_index: int = 0,
) -> torch.Tensor:
    """Convert one packed embedding batch into normalized tensors."""
    if use_variants:
        embedding = torch.tensor(batch_embeddings, dtype=torch.float32)
    else:
        embedding = torch.tensor(
            batch_embeddings[:, original_variant_index, :],
            dtype=torch.float32,
        )

    if normalize:
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

    return embedding


def predict_single_batch(
    model: torch.nn.Module,
    prototypes: torch.Tensor,
    embeddings: torch.Tensor,
    device: torch.device,
    scoring_temperature: float = 1.0,
) -> np.ndarray:
    """Generate probabilities for one batch of embeddings."""
    model.eval()

    with torch.no_grad():
        embeddings = embeddings.to(device)
        outputs = model(embeddings)

        if torch.is_tensor(outputs):
            embeddings_out = outputs
        elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            _, embeddings_out = outputs
        else:
            raise ValueError(f"Unexpected model output format: {type(outputs)}")

        probs = compute_prototype_probabilities(
            embeddings=embeddings_out,
            prototypes=prototypes,
            scoring_temperature=scoring_temperature,
            logger=None,
        )
        return probs.cpu().numpy()


def run_inference(
    embedding_dir: Path,
    model_dir: Path,
    config: ConfigDict,
    is_single_stage: bool,
    device: torch.device,
    pooling_type: str,
    batch_size: int,
    logger: logging.Logger,
) -> Tuple[List[str], np.ndarray]:
    """Run ensemble inference across all fold checkpoints."""
    packed_embeddings, sequence_ids, metadata = open_packed_embedding_dataset(
        embedding_dir
    )
    if metadata.get("pooling_type") != pooling_type:
        raise ValueError(
            "Packed embedding dataset pooling_type does not match request: "
            f"expected {pooling_type!r}, got {metadata.get('pooling_type')!r}"
        )
    if len(sequence_ids) == 0:
        raise ValueError(f"No packed embeddings found in {embedding_dir}")
    original_variant_index = int(metadata.get("original_variant_index", 0))
    if not 0 <= original_variant_index < packed_embeddings.shape[1]:
        raise ValueError(
            "Packed embedding dataset original_variant_index is out of bounds"
        )

    logger.info("Found %s packed embeddings", len(sequence_ids))

    num_folds = config.training.num_folds
    models = []
    prototypes_list = []
    scoring_temperatures = []

    logger.info("Loading models from %s folds...", num_folds)
    for fold in range(1, num_folds + 1):
        model_path = get_checkpoint_path(model_dir, fold, is_single_stage)

        if model_path.exists():
            logger.info("  Loading fold %s from %s", fold, model_path)
            model, prototypes, scoring_temp = load_prototype_ranking_model(
                model_path,
                config,
                device,
                is_single_stage,
                logger,
            )
            if prototypes is None:
                raise ValueError(
                    f"No prototypes found in checkpoint {model_path}. "
                    "This model may not be trained with prototype ranking."
                )
            models.append(model)
            prototypes_list.append(prototypes)
            scoring_temperatures.append(scoring_temp)
        else:
            logger.warning("  Model not found for fold %s: %s", fold, model_path)

    if not models:
        raise ValueError("No models loaded! Check model paths.")

    logger.info("Successfully loaded %s models", len(models))

    normalize = getattr(config.features, "normalize", True)
    use_variants = getattr(config.training, "use_variants", False)

    logger.info("Embedding configuration:")
    logger.info("  Normalize: %s", normalize)
    logger.info("  Use variants: %s", use_variants)
    logger.info("  Pooling type: %s", pooling_type)
    logger.info("  Original variant index: %s", original_variant_index)

    all_predictions = []

    logger.info("Running inference...")
    for start_idx in tqdm(
        range(0, len(sequence_ids), batch_size),
        desc="Processing batches",
    ):
        batch_embeddings = packed_embeddings[start_idx : start_idx + batch_size]
        stacked_embeddings = load_embedding_batch(
            batch_embeddings,
            normalize=normalize,
            use_variants=use_variants,
            original_variant_index=original_variant_index,
        )
        fold_predictions = []
        for model, prototypes, scoring_temp in zip(
            models,
            prototypes_list,
            scoring_temperatures,
            strict=True,
        ):
            fold_predictions.append(
                predict_single_batch(
                    model,
                    prototypes,
                    stacked_embeddings,
                    device,
                    scoring_temp,
                )
            )

        ensemble_preds = np.mean(fold_predictions, axis=0)
        all_predictions.extend(ensemble_preds)

    predictions = np.array(all_predictions)
    logger.info("Inference completed. Generated %s predictions", len(predictions))
    logger.info(
        "Prediction statistics: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        predictions.min(),
        predictions.max(),
        predictions.mean(),
        np.median(predictions),
    )

    return sequence_ids, predictions


def save_predictions(
    sequence_ids: List[str],
    predictions: np.ndarray,
    binary_labels: np.ndarray,
    output_file: Path,
    threshold: float,
    logger: logging.Logger,
) -> None:
    """Save inference predictions to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sequence_id", "probability", "binary_prediction", "threshold"]
        )
        for seq_id, prob, label in zip(
            sequence_ids,
            predictions,
            binary_labels,
            strict=True,
        ):
            writer.writerow([seq_id, f"{prob:.6f}", int(label), threshold])

    logger.info("Predictions saved to %s", output_file)

    num_positive = int(np.sum(binary_labels))
    num_negative = len(binary_labels) - num_positive
    logger.info("Prediction summary:")
    logger.info("  Total sequences: %s", len(sequence_ids))
    logger.info(
        "  Predicted positive (label=1): %s (%.2f%%)",
        num_positive,
        num_positive / len(binary_labels) * 100,
    )
    logger.info(
        "  Predicted negative (label=0): %s (%.2f%%)",
        num_negative,
        num_negative / len(binary_labels) * 100,
    )


def main() -> None:
    """Run prototype ranking inference from a saved training run directory."""
    args = parse_inference_args()

    if not args.embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {args.embedding_dir}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    if args.output_file is None:
        args.output_file = args.model_dir / "predictions.csv"

    logger = setup_logger(
        name="prototype_ranking_inference",
        output_dir=args.model_dir,
        log_file_name="inference",
    )

    try:
        config_path = args.model_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_run_config(args.model_dir)
        logger.info("Loaded configuration from %s", config_path)

        device = resolve_device(config)
        logger.info("Using device: %s", device)

        require_prototype_model_type(config)
        logger.info("Model type: SimplePredictor")
        logger.info(
            "Training type: %s",
            "single-stage" if args.single_stage else "two-stage",
        )

        sequence_ids, predictions = run_inference(
            embedding_dir=args.embedding_dir,
            model_dir=args.model_dir,
            config=config,
            is_single_stage=args.single_stage,
            device=device,
            pooling_type=args.pooling_type,
            batch_size=args.batch_size,
            logger=logger,
        )

        binary_labels = (predictions >= args.threshold).astype(np.int32)
        logger.info("Applied classification threshold: %s", args.threshold)

        save_predictions(
            sequence_ids=sequence_ids,
            predictions=predictions,
            binary_labels=binary_labels,
            output_file=args.output_file,
            threshold=args.threshold,
            logger=logger,
        )

        logger.info("Inference completed successfully!")
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
