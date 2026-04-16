# ruff: noqa: E501,I001,B007,B905,F541,F841
#!/usr/bin/env python3
"""
Analysis of prototype ranking model outcomes.

This script provides deeper insights into:
1. Embedding space visualization using dimensionality reduction
2. Detailed prototype location analysis
3. Advanced distance score analysis with confidence intervals
4. Sample-to-prototype distance analysis
5. Cross-fold consistency analysis
6. Embedding space alignment metrics (SAD, SAA, CAD, CAC, GPU, PEU)
"""

import argparse
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from ml_collections import ConfigDict
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from effector_bincls.checkpoints import get_checkpoint_path
from effector_bincls.data import (
    DEFAULT_PARTITION_COLUMN,
    SimpleDataset,
    load_labeled_dataset,
    resolve_label_columns,
)
from effector_bincls.metrics import find_optimal_threshold
from effector_bincls.prototype_loading import load_prototype_ranking_model
from effector_bincls.analysis.alignment import calculate_all_alignment_metrics
from effector_bincls.plotting import (
    plot_threshold_analysis as plot_threshold_analysis_comprehensive,
)
from effector_bincls.run_utils import load_config

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PrototypeAnalyzer:
    """
    Analyzer for prototype ranking model outcomes with embedding visualization.
    """

    # Prototype index convention aligned with training/scoring utilities:
    # Index 0 -> negative prototype (class 0)
    # Index 1 -> positive prototype (class 1)
    NEG_IDX = 0
    POS_IDX = 1

    def __init__(self, run_dir: Path, config_path: Optional[Path] = None):
        """
        Args:
            run_dir: Path to the training run directory
            config_path: Optional path to config file
        """
        self.run_dir = Path(run_dir)
        self.config_path = config_path or (self.run_dir / "config.yml")

        # Load OOF predictions
        self.oof_predictions_file = self.run_dir / "oof_predictions.npz"
        self.load_oof_predictions()

        # Load config
        self.load_config()

        # Initialize results storage
        self.analysis_results = {}

    def load_oof_predictions(self):
        """Load out-of-fold predictions."""
        if not self.oof_predictions_file.exists():
            raise FileNotFoundError(
                f"OOF predictions file not found: {self.oof_predictions_file}"
            )

        data = np.load(self.oof_predictions_file, allow_pickle=True)
        self.predictions = data["predictions"].item()
        self.labels = data["labels"].item()

        logger.info(f"Loaded OOF predictions for {len(self.predictions)} folds")

    def load_config(self):
        """Load training configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            self.config = {}
            return

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        logger.info("Loaded training configuration")

    def load_prototypes_from_checkpoints(
        self, is_single_stage: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Load prototypes from checkpoint files for all folds.

        Args:
            is_single_stage: True for single-stage training (fold_X/checkpoint.pt),
                         False for two-stage (fold_X/finetuning/checkpoint.pt)

        Returns:
            Dictionary mapping fold number to prototypes array
        """
        logger.info("Loading prototypes from checkpoint files...")

        prototypes_by_fold = {}

        for fold in sorted(self.predictions.keys()):
            checkpoint_path = get_checkpoint_path(
                self.run_dir, fold, is_single_stage=is_single_stage
            )

            if not checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint not found for fold {fold}: {checkpoint_path}"
                )
                continue

            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )

                if "prototypes" in checkpoint:
                    prototypes = checkpoint["prototypes"].numpy()
                    prototypes_by_fold[fold] = prototypes
                    logger.info(
                        f"Loaded prototypes for fold {fold}: shape {prototypes.shape}"
                    )

                    # Verify [p, -p] pattern
                    if prototypes.shape[0] == 2:
                        diff = np.linalg.norm(prototypes[0] + prototypes[1])
                        logger.info(
                            f"Fold {fold} prototype symmetry check: {diff:.6f} (should be ~0)"
                        )

                        # Log prototype norms
                        norms = np.linalg.norm(prototypes, axis=1)
                        logger.info(
                            f"Fold {fold} prototype norms: [{norms[0]:.3f}, {norms[1]:.3f}]"
                        )
                else:
                    logger.warning(f"No prototypes found in checkpoint for fold {fold}")

            except Exception as e:
                logger.error(f"Error loading prototypes for fold {fold}: {e}")

        logger.info(
            f"Successfully loaded prototypes for {len(prototypes_by_fold)} folds"
        )
        return prototypes_by_fold

    def generate_embeddings_with_prototypes(
        self,
        prototypes_by_fold: Dict[int, np.ndarray],
        fold: int = 1,
        sample_size: int = 100,
        is_single_stage: bool = False,
    ) -> Dict:
        """
        Generate embeddings by loading trained models and running inference on OOF data.
        Then apply dimensionality reduction to both embeddings and prototypes.

        This method focuses on:
        1. Loading a trained model and generating embeddings (with multi-view support)
        2. Processing embeddings consistently with training pipeline (averaging across views)
        3. Applying dimensionality reduction to both embeddings and prototypes together
        4. Preserving original multi-view embeddings for alignment metrics

        Args:
            prototypes_by_fold: Dictionary mapping fold to prototypes (already loaded by load_prototypes_from_checkpoints)
            fold: Fold number to use for model loading and inference (default: 1)
            sample_size: Number of samples to use for visualization (default: 100)
            is_single_stage: True for single-stage training (fold_X/checkpoint.pt),
                         False for two-stage (fold_X/finetuning/checkpoint.pt)

        Returns:
            Dictionary containing:
            - embeddings_nd: Multi-view embeddings (N x num_variants x D) for all analysis
            - embeddings_2d: 2D projections for visualization (N x 2) - from averaged embeddings
            - labels: Labels for the sampled data (N,)
            - prototypes_nd: Original prototype vectors (2 x D), index 0 = negative, 1 = positive
            - prototypes_2d: Projected prototype vectors (2 x 2)
            - prototype_distance_2d: L2 distance between projected prototypes (float)
            - scoring_temperature: Temperature used for prototype scoring
        """
        logger.info(
            f"Generating embeddings using trained models (fold {fold}, sample_size {sample_size})..."
        )
        config_path = self.run_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = ConfigDict(load_config(config_path))

        # Set device
        device = torch.device(
            f"cuda:{config.hardware.gpu_id}"
            if config.hardware.gpu_id >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        # Get all test sequence IDs for sampling
        test_csv_value = getattr(config.data, "finetuning_csv_path", None)
        if test_csv_value is None:
            test_csv_value = config.data.csv_path
        test_csv_path = Path(test_csv_value)
        label_config = getattr(config.data, "label_config", {})
        sequence_id_column, _ = resolve_label_columns(label_config)
        df = load_labeled_dataset(
            test_csv_path,
            label_config=label_config,
            required_partitions={"test"},
        )
        test_df = df[df[DEFAULT_PARTITION_COLUMN] == "test"]

        if len(test_df) == 0:
            raise ValueError("No test data found in CSV")

        all_test_ids = test_df[sequence_id_column].tolist()
        logger.info(f"Found {len(all_test_ids)} test sequences")

        # For visualization, we'll sample a subset to avoid memory issues
        if sample_size == -1:
            # Use all available test samples
            sampled_ids = all_test_ids
            logger.info(
                f"Using all {len(sampled_ids)} test sequences for visualization"
            )
        else:
            # Use provided sample_size, but don't exceed available data
            sample_size = min(sample_size, len(all_test_ids))
            sampled_ids = random.sample(all_test_ids, sample_size)
            logger.info(f"Sampling {sample_size} test sequences for visualization")

        model_type = config.model.type.lower()
        if model_type not in ["simple_predictor", "simple"]:
            raise ValueError(
                f"Unsupported model type '{model_type}' for prototype analysis."
            )

        dataset = SimpleDataset(
            embedding_dir=config.data.embedding_dir,
            csv_path=str(test_csv_path),
            sequence_ids=sampled_ids,
            normalize=getattr(config.features, "normalize", True),
            pooling_type=getattr(config.features, "pooling_type", "mean"),
            use_variants=getattr(config.training, "use_variants", False),
            label_config=label_config,
            logger=logger,
        )

        # Create data loader
        from torch.utils.data import DataLoader

        num_workers = int(getattr(config.hardware, "num_workers", 0))
        data_loader = DataLoader(
            dataset,
            batch_size=64,  # Smaller batch size for visualization
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        # Run inference using the specified fold
        if fold not in prototypes_by_fold:
            raise ValueError(f"Fold {fold} not available for inference")

        # Load the trained model for this fold using the existing infrastructure
        model_path = get_checkpoint_path(
            self.run_dir, fold, is_single_stage=is_single_stage
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for fold {fold}: {model_path}")

        # Use the existing model loading function
        model, _, scoring_temperature = load_prototype_ranking_model(
            model_path, config, device, is_single_stage
        )

        # Set model to evaluation mode
        model.eval()
        logger.info(f"Successfully loaded model checkpoint from {model_path}")
        logger.info(
            f"Model type: {type(model).__name__}, Scoring temperature: {scoring_temperature:.3f}"
        )

        # Use prototypes already loaded by load_prototypes_from_checkpoints
        prototypes_nd_tensor = torch.tensor(prototypes_by_fold[fold], device=device)
        logger.info(
            f"Using prototypes from fold {fold}, shape: {prototypes_nd_tensor.shape}"
        )

        # Validate prototype shape and values
        if prototypes_nd_tensor.shape[0] != 2:
            raise ValueError(
                f"Expected 2 prototypes, got {prototypes_nd_tensor.shape[0]}"
            )

        # Check prototype separation in original space
        proto_similarity = torch.cosine_similarity(
            prototypes_nd_tensor[self.POS_IDX],
            prototypes_nd_tensor[self.NEG_IDX],
            dim=0,
        ).item()
        logger.info(f"Prototype similarity in original space: {proto_similarity:.3f}")

        logger.info(
            f"Running inference for {sample_size} samples using fold {fold} model..."
        )

        # Run inference to get embeddings
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                if isinstance(features, tuple):
                    features = tuple(f.to(device) for f in features)
                    outputs = model(*features)
                else:
                    features = features.to(device)
                    outputs = model(features)

                # Since load_prototype_ranking_model sets the model to pretraining mode,
                # the model should return embeddings directly
                if torch.is_tensor(outputs):
                    embeddings = outputs
                    logger.debug(
                        f"Model returned embeddings directly: {embeddings.shape}"
                    )
                elif isinstance(outputs, (tuple, list)):
                    if len(outputs) == 2:
                        # (logits, embeddings) format - use embeddings for prototype ranking
                        logits, embeddings = outputs
                        logger.debug(
                            f"Using embeddings from tuple output: {embeddings.shape}"
                        )
                    else:
                        logger.warning(
                            f"Unexpected tuple output length: {len(outputs)}"
                        )
                        continue
                else:
                    logger.warning(f"Unexpected model output format: {type(outputs)}")
                    continue

                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(1)
                    logger.debug(
                        "Promoted single-view embeddings to singleton-view analysis shape: %s",
                        embeddings.shape,
                    )
                elif embeddings.dim() != 3:
                    raise ValueError(
                        "Expected embeddings with shape [batch_size, feat_dim] or "
                        f"[batch_size, num_variants, feat_dim], got {embeddings.shape}."
                    )

                num_variants = embeddings.shape[1]
                if num_variants < 2:
                    logger.info(
                        "Prototype analysis running in single-view mode for this batch"
                    )
                else:
                    logger.debug(
                        "Processing multi-view embeddings: %s (%s variants)",
                        embeddings.shape,
                        num_variants,
                    )

                # Store multi-view embeddings and labels
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_embeddings:
            raise ValueError("No embeddings could be generated from the model")

        # Combine multi-view embeddings and labels
        embeddings = np.concatenate(
            all_embeddings, axis=0
        )  # [N, num_variants, feat_dim]
        labels = np.concatenate(all_labels, axis=0)

        # Flatten labels if they're 2D
        if labels.ndim > 1:
            labels = labels.flatten()

        logger.info(
            "Generated %s embedding samples, shape: %s",
            len(embeddings),
            embeddings.shape,
        )
        logger.info(f"Labels shape: {labels.shape}")

        # Validate multi-view structure
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected 3D multi-view embeddings [N, num_variants, feat_dim], got shape {embeddings.shape}"
            )

        # Convert prototypes to numpy for consistency
        prototypes_np = prototypes_nd_tensor.cpu().numpy()
        logger.info(f"Using prototype vectors, shape: {prototypes_np.shape}")

        # Apply dimensionality reduction to both embeddings and prototypes
        logger.info("Applying UMAP dimensionality reduction...")

        # For visualization, use averaged embeddings across variants (consistent with training)
        embeddings_averaged = embeddings.mean(axis=1)  # [N, feat_dim]
        logger.info(
            f"Averaged embeddings for visualization: {embeddings_averaged.shape}"
        )

        # Combine averaged embeddings and actual prototype vectors for consistent UMAP fitting
        # This ensures both embeddings and prototypes are projected to the same 2D space
        combined_data = np.vstack([embeddings_averaged, prototypes_np])

        # Check for any infinite or NaN values and clean data thoroughly
        if not np.isfinite(combined_data).all():
            logger.warning("Found non-finite values in data, replacing with zeros")
            combined_data = np.nan_to_num(
                combined_data, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Additional data validation for UMAP compatibility
        if combined_data.shape[0] < 15:  # UMAP needs at least n_neighbors samples
            logger.warning("Too few samples for UMAP, using PCA instead")
            use_umap = False
        else:
            use_umap = True

        # Try UMAP, fallback to PCA if it fails
        if use_umap:
            try:
                # Fit UMAP on combined data
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(
                        15, combined_data.shape[0] - 1
                    ),  # Ensure n_neighbors <= n_samples
                    min_dist=0.1,
                    random_state=42,
                    metric="cosine",
                )

                # Fit on combined data and transform
                combined_2d = reducer.fit_transform(combined_data)
                logger.info("Successfully applied UMAP dimensionality reduction")

            except Exception as e:
                logger.warning(f"UMAP failed: {e}, falling back to PCA")
                use_umap = False

        if not use_umap:
            # Fallback to PCA
            pca = PCA(n_components=2, random_state=42)
            combined_2d = pca.fit_transform(combined_data)
            logger.info(
                f"Applied PCA dimensionality reduction, explained variance: {pca.explained_variance_ratio_.sum():.3f}"
            )

        # Split back into embeddings and prototypes using the correct indices
        # The first len(embeddings_averaged) vectors are the averaged embeddings, the last 2 are the prototypes
        num_embeddings = len(embeddings_averaged)
        embeddings_2d = combined_2d[:num_embeddings]
        prototypes_2d = combined_2d[num_embeddings:]

        # Validate that we have the correct number of prototype vectors
        if len(prototypes_2d) != 2:
            raise ValueError(
                f"Expected 2 prototype vectors in 2D space, got {len(prototypes_2d)}"
            )

        # Build clean projection bundle
        projection = {
            "embeddings_nd": embeddings,  # Multi-view embeddings [N, num_variants, feat_dim]
            "embeddings_2d": embeddings_2d,  # 2D visualization (from averaged embeddings)
            "labels": labels,
            "prototypes_nd": prototypes_np,
            "prototypes_2d": prototypes_2d,
            "prototype_distance_2d": float(
                np.linalg.norm(
                    prototypes_2d[self.POS_IDX] - prototypes_2d[self.NEG_IDX]
                )
            ),
            "scoring_temperature": float(scoring_temperature),
        }

        logger.info(
            f"UMAP projection complete. Embeddings shape: {embeddings_2d.shape}"
        )
        logger.info(
            f"Prototype separation in 2D: {projection['prototype_distance_2d']:.3f}"
        )

        return projection

    def visualize_embedding_space_with_prototypes(
        self,
        projection: Dict,
        save_dir: Path,
    ):
        """
        Visualize the embedding space using UMAP projections and prototypes.

        This method uses the precomputed projection bundle from
        generate_embeddings_with_prototypes.

        Args:
            projection: Dictionary containing embeddings/prototypes in 2D and original space
            save_dir: Directory to save visualizations
        """
        logger.info(
            "Creating embedding space visualizations with prototypes and UMAP..."
        )

        embeddings_2d = projection["embeddings_2d"]
        labels = projection["labels"]
        prototypes_2d = projection["prototypes_2d"]

        # Create multiple visualization approaches
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Extract prototype positions from projection
        pos_prototype_2d = prototypes_2d[self.POS_IDX]
        neg_prototype_2d = prototypes_2d[self.NEG_IDX]

        # 1. Scatter plot with class colors and prototypes
        pos_mask = labels == 1
        neg_mask = labels == 0

        axes[0, 0].scatter(
            embeddings_2d[pos_mask, 0],
            embeddings_2d[pos_mask, 1],
            alpha=0.6,
            color="green",
            label="Positive",
            s=20,
        )
        axes[0, 0].scatter(
            embeddings_2d[neg_mask, 0],
            embeddings_2d[neg_mask, 1],
            alpha=0.6,
            color="red",
            label="Negative",
            s=20,
        )

        # Plot prototypes
        axes[0, 0].scatter(
            pos_prototype_2d[0],
            pos_prototype_2d[1],
            color="darkgreen",
            s=400,
            marker="*",
            label="Positive Prototype",
            edgecolors="black",
        )
        axes[0, 0].scatter(
            neg_prototype_2d[0],
            neg_prototype_2d[1],
            color="darkred",
            s=400,
            marker="*",
            label="Negative Prototype",
            edgecolors="black",
        )

        axes[0, 0].set_title("UMAP Embedding Space - prototypes")
        axes[0, 0].set_xlabel("UMAP Component 1")
        axes[0, 0].set_ylabel("UMAP Component 2")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Density plot
        from scipy.stats import gaussian_kde

        # Define the mesh grid for density plots
        x_range = np.linspace(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max(), 100)
        y_range = np.linspace(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Positive samples density
        if np.sum(pos_mask) > 1:
            pos_points = embeddings_2d[pos_mask]
            try:
                kde_pos = gaussian_kde(pos_points.T)
                Z_pos = kde_pos(positions).reshape(X.shape)
                axes[0, 1].contour(X, Y, Z_pos, colors="green", alpha=0.7, levels=5)
            except Exception as e:
                logger.warning(f"KDE failed for positive samples: {e}")
                # Fallback to simple scatter
                axes[0, 1].scatter(
                    pos_points[:, 0],
                    pos_points[:, 1],
                    alpha=0.7,
                    color="green",
                    s=30,
                    label="Positive",
                )

        # Negative samples density
        if np.sum(neg_mask) > 1:
            neg_points = embeddings_2d[neg_mask]
            try:
                kde_neg = gaussian_kde(neg_points.T)
                Z_neg = kde_neg(positions).reshape(X.shape)
                axes[0, 1].contour(X, Y, Z_neg, colors="red", alpha=0.7, levels=5)
            except Exception as e:
                logger.warning(f"KDE failed for negative samples: {e}")
                # Fallback to simple scatter
                axes[0, 1].scatter(
                    neg_points[:, 0],
                    neg_points[:, 1],
                    alpha=0.7,
                    color="red",
                    s=30,
                    label="Negative",
                )

        # Plot prototypes
        axes[0, 1].scatter(
            pos_prototype_2d[0],
            pos_prototype_2d[1],
            color="darkgreen",
            s=400,
            marker="*",
            edgecolors="black",
        )
        axes[0, 1].scatter(
            neg_prototype_2d[0],
            neg_prototype_2d[1],
            color="darkred",
            s=400,
            marker="*",
            edgecolors="black",
        )

        axes[0, 1].set_title("Density Plot with prototypes")
        axes[0, 1].set_xlabel("UMAP Component 1")
        axes[0, 1].set_ylabel("UMAP Component 2")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distance-based visualization
        # Calculate distances from each point to prototypes
        # Combine embeddings and prototypes for distance calculation
        all_points = np.vstack(
            [
                embeddings_2d,
                pos_prototype_2d.reshape(1, -1),
                neg_prototype_2d.reshape(1, -1),
            ]
        )
        distances = euclidean_distances(all_points)

        # Distance to positive prototype (second to last point)
        dist_to_pos = distances[:-2, -2]
        # Distance to negative prototype (last point)
        dist_to_neg = distances[:-2, -1]

        # Color points based on relative distance to prototypes
        relative_dist = dist_to_pos - dist_to_neg
        scatter = axes[0, 2].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=relative_dist,
            cmap="RdBu_r",
            alpha=0.7,
            s=30,
        )

        # Plot prototypes
        axes[0, 2].scatter(
            pos_prototype_2d[0],
            pos_prototype_2d[1],
            color="darkgreen",
            s=400,
            marker="*",
            label="Positive Prototype",
            edgecolors="black",
        )
        axes[0, 2].scatter(
            neg_prototype_2d[0],
            neg_prototype_2d[1],
            color="darkred",
            s=400,
            marker="*",
            label="Negative Prototype",
            edgecolors="black",
        )

        axes[0, 2].set_title("Distance to Prototypes")
        axes[0, 2].set_xlabel("UMAP Component 1")
        axes[0, 2].set_ylabel("UMAP Component 2")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label="Distance to Pos - Distance to Neg")

        # 4. Class separation analysis
        # Calculate separation metrics
        pos_points = embeddings_2d[pos_mask]
        neg_points = embeddings_2d[neg_mask]

        if len(pos_points) > 0 and len(neg_points) > 0:
            # Calculate centroids
            pos_centroid = np.mean(pos_points, axis=0)
            neg_centroid = np.mean(neg_points, axis=0)

            # Plot centroids
            axes[1, 0].scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap="RdYlGn",
                alpha=0.6,
                s=20,
            )
            axes[1, 0].scatter(
                pos_centroid[0],
                pos_centroid[1],
                color="darkgreen",
                s=150,
                marker="o",
                label="Positive Centroid",
                edgecolors="black",
            )
            axes[1, 0].scatter(
                neg_centroid[0],
                neg_centroid[1],
                color="darkred",
                s=150,
                marker="o",
                label="Negative Centroid",
                edgecolors="black",
            )

            # Plot learned prototypes
            axes[1, 0].scatter(
                pos_prototype_2d[0],
                pos_prototype_2d[1],
                color="darkgreen",
                s=400,
                marker="*",
                label="Positive Prototype",
                edgecolors="white",
                linewidth=2,
            )
            axes[1, 0].scatter(
                neg_prototype_2d[0],
                neg_prototype_2d[1],
                color="darkred",
                s=400,
                marker="*",
                label="Negative Prototype",
                edgecolors="white",
                linewidth=2,
            )

            axes[1, 0].set_title("Class Centroids vs Learned Prototypes")
            axes[1, 0].set_xlabel("UMAP Component 1")
            axes[1, 0].set_ylabel("UMAP Component 2")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Summary statistics
        axes[1, 2].axis("off")
        stats_text = f"""
        Embedding Space Analysis
        
        Total samples: {len(embeddings_2d)}
        Positive samples: {np.sum(pos_mask)}
        Negative samples: {np.sum(neg_mask)}
        
        Prototype separation (2D): {projection["prototype_distance_2d"]:.3f}
        
        Distance Analysis:
        - Avg dist to pos proto: {np.mean(dist_to_pos):.3f}
        - Avg dist to neg proto: {np.mean(dist_to_neg):.3f}
        - Pos samples closer to pos: {np.mean(dist_to_pos[pos_mask] < dist_to_neg[pos_mask]):.1%}
        - Neg samples closer to neg: {np.mean(dist_to_neg[neg_mask] < dist_to_pos[neg_mask]):.1%}
        

        """
        axes[1, 2].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        save_path = save_dir / "embedding_space_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Embedding space visualization saved to {save_path}")

    def analyze_prototype_distances(self, projection: Dict) -> Dict:
        """
        Analyze similarities between samples and prototypes using original embeddings.

        This method computes distances in the original embedding space (not 2D projections)
        to match how prototype ranking actually works during training and inference.

        Args:
            projection: Dictionary containing original and 2D embeddings and prototypes

        Returns:
            Dictionary with distance analysis results
        """
        logger.info(
            "Analyzing sample-to-prototype similarities using multi-view embeddings..."
        )

        multiview_embeddings = projection[
            "embeddings_nd"
        ]  # [N, num_variants, feat_dim]
        labels = projection["labels"]
        prototypes = projection["prototypes_nd"]

        # Average across variants for prototype scoring (consistent with training)
        averaged_embeddings = multiview_embeddings.mean(axis=1)  # [N, feat_dim]
        logger.info(
            f"Using averaged embeddings for prototype scoring: {averaged_embeddings.shape}"
        )

        # FIXED: Calculate cosine similarity (not Euclidean distance) - this is how prototype ranking works
        # This matches the logic in compute_prototype_distance_scores utility function
        # Calculate similarities to prototypes using cosine similarity
        # This matches how the model computes distance scores during training
        pos_similarities = cosine_similarity(
            averaged_embeddings,
            prototypes[self.POS_IDX : self.POS_IDX + 1],
        )[:, 0]
        neg_similarities = cosine_similarity(
            averaged_embeddings,
            prototypes[self.NEG_IDX : self.NEG_IDX + 1],
        )[:, 0]

        # FIXED: Compute distance scores exactly as in compute_prototype_distance_scores
        # This matches the logic: distance_scores = similarities[:, 1] - similarities[:, 0]
        distance_scores = pos_similarities - neg_similarities

        # Separate by class for analysis
        pos_samples_pos_sim = pos_similarities[labels == 1]
        pos_samples_neg_sim = neg_similarities[labels == 1]
        neg_samples_pos_sim = pos_similarities[labels == 0]
        neg_samples_neg_sim = neg_similarities[labels == 0]

        # Calculate distance scores by class
        pos_samples_ranking = distance_scores[labels == 1]
        neg_samples_ranking = distance_scores[labels == 0]

        # Calculate statistics
        # Apply scoring temperature scaling exactly as in training when computing distance scores
        scoring_temperature = float(projection.get("scoring_temperature", 1.0))

        analysis = {
            "pos_samples_to_pos_proto": {
                "mean": np.mean(pos_samples_pos_sim),
                "std": np.std(pos_samples_pos_sim),
                "median": np.median(pos_samples_pos_sim),
            },
            "pos_samples_to_neg_proto": {
                "mean": np.mean(pos_samples_neg_sim),
                "std": np.std(pos_samples_neg_sim),
                "median": np.median(pos_samples_neg_sim),
            },
            "neg_samples_to_pos_proto": {
                "mean": np.mean(neg_samples_pos_sim),
                "std": np.std(neg_samples_pos_sim),
                "median": np.median(neg_samples_pos_sim),
            },
            "neg_samples_to_neg_proto": {
                "mean": np.mean(neg_samples_neg_sim),
                "std": np.std(neg_samples_neg_sim),
                "median": np.median(neg_samples_neg_sim),
            },
            "distance_scores": {
                "pos_samples_mean": float(
                    np.mean(pos_samples_ranking / scoring_temperature)
                ),
                "pos_samples_std": float(
                    np.std(pos_samples_ranking / scoring_temperature)
                ),
                "neg_samples_mean": float(
                    np.mean(neg_samples_ranking / scoring_temperature)
                ),
                "neg_samples_std": float(
                    np.std(neg_samples_ranking / scoring_temperature)
                ),
                "overall_mean": float(np.mean(distance_scores / scoring_temperature)),
                "overall_std": float(np.std(distance_scores / scoring_temperature)),
                "scoring_temperature": scoring_temperature,
            },
            "prototype_separation_2d": projection["prototype_distance_2d"],
            "pos_closer_to_pos": np.mean(
                pos_samples_pos_sim > pos_samples_neg_sim
            ),  # Higher similarity = closer
            "neg_closer_to_neg": np.mean(
                neg_samples_neg_sim > neg_samples_pos_sim
            ),  # Higher similarity = closer
        }

        # FIXED: Calculate ranking accuracy properly by checking each class separately
        # This avoids the shape mismatch issue
        pos_correct = (
            np.sum(pos_samples_ranking > 0) if len(pos_samples_ranking) > 0 else 0
        )
        neg_correct = (
            np.sum(neg_samples_ranking < 0) if len(neg_samples_ranking) > 0 else 0
        )
        total_samples = len(pos_samples_ranking) + len(neg_samples_ranking)

        if total_samples > 0:
            ranking_accuracy = (pos_correct + neg_correct) / total_samples
        else:
            ranking_accuracy = 0.0

        analysis["ranking_accuracy"] = ranking_accuracy

        logger.info(
            f"Positive samples more similar to positive prototype: {analysis['pos_closer_to_pos']:.1%}"
        )
        logger.info(
            f"Negative samples more similar to negative prototype: {analysis['neg_closer_to_neg']:.1%}"
        )
        logger.info(
            f"Ranking accuracy (correct direction): {analysis['ranking_accuracy']:.1%}"
        )
        logger.info(
            f"Prototype separation (2D): {analysis['prototype_separation_2d']:.3f}"
        )

        # Log similarity statistics
        logger.info(
            f"Positive samples - Pos proto similarity: {analysis['pos_samples_to_pos_proto']['mean']:.3f} ± {analysis['pos_samples_to_pos_proto']['std']:.3f}"
        )
        logger.info(
            f"Positive samples - Neg proto similarity: {analysis['pos_samples_to_neg_proto']['mean']:.3f} ± {analysis['pos_samples_to_neg_proto']['std']:.3f}"
        )
        logger.info(
            f"Negative samples - Pos proto similarity: {analysis['neg_samples_to_pos_proto']['mean']:.3f} ± {analysis['neg_samples_to_pos_proto']['std']:.3f}"
        )
        logger.info(
            f"Negative samples - Neg proto similarity: {analysis['neg_samples_to_neg_proto']['mean']:.3f} ± {analysis['neg_samples_to_neg_proto']['std']:.3f}"
        )

        # Log distance score statistics
        logger.info(
            f"Positive samples distance score: {analysis['distance_scores']['pos_samples_mean']:.3f} ± {analysis['distance_scores']['pos_samples_std']:.3f}"
        )
        logger.info(
            f"Negative samples distance score: {analysis['distance_scores']['neg_samples_mean']:.3f} ± {analysis['distance_scores']['neg_samples_std']:.3f}"
        )
        logger.info(
            f"Overall distance score: {analysis['distance_scores']['overall_mean']:.3f} ± {analysis['distance_scores']['overall_std']:.3f}"
        )

        return analysis

    def analyze_prototype_consistency(
        self, prototypes_by_fold: Dict[int, np.ndarray]
    ) -> Dict:
        """Analyze consistency of prototypes across folds."""
        logger.info("Analyzing prototype consistency across folds...")

        if len(prototypes_by_fold) < 2:
            logger.warning("Need at least 2 folds to analyze prototype consistency")
            return {}

        # Extract positive prototypes (first row of each prototype matrix)
        pos_prototypes = np.array(
            [protos[self.POS_IDX] for protos in prototypes_by_fold.values()]
        )
        neg_prototypes = np.array(
            [protos[self.NEG_IDX] for protos in prototypes_by_fold.values()]
        )

        # Calculate pairwise similarities
        pos_similarities = []
        neg_similarities = []
        symmetry_errors = []

        for i in range(len(pos_prototypes)):
            for j in range(i + 1, len(pos_prototypes)):
                # Positive prototype similarities
                pos_sim = cosine_similarity(
                    pos_prototypes[i].reshape(1, -1), pos_prototypes[j].reshape(1, -1)
                )[0, 0]
                pos_similarities.append(pos_sim)

                # Negative prototype similarities
                neg_sim = cosine_similarity(
                    neg_prototypes[i].reshape(1, -1), neg_prototypes[j].reshape(1, -1)
                )[0, 0]
                neg_similarities.append(neg_sim)

                # Check symmetry (p, -p) pattern
                symmetry_error = np.linalg.norm(pos_prototypes[i] + neg_prototypes[i])
                symmetry_errors.append(symmetry_error)

        analysis = {
            "pos_prototype_consistency": {
                "mean_similarity": np.mean(pos_similarities),
                "std_similarity": np.std(pos_similarities),
                "min_similarity": np.min(pos_similarities),
                "max_similarity": np.max(pos_similarities),
            },
            "neg_prototype_consistency": {
                "mean_similarity": np.mean(neg_similarities),
                "std_similarity": np.std(neg_similarities),
                "min_similarity": np.min(neg_similarities),
                "max_similarity": np.max(neg_similarities),
            },
            "symmetry_analysis": {
                "mean_symmetry_error": np.mean(symmetry_errors),
                "std_symmetry_error": np.std(symmetry_errors),
                "max_symmetry_error": np.max(symmetry_errors),
            },
            "prototype_norms": {
                "pos_mean_norm": np.mean([np.linalg.norm(p) for p in pos_prototypes]),
                "pos_std_norm": np.std([np.linalg.norm(p) for p in pos_prototypes]),
                "neg_mean_norm": np.mean([np.linalg.norm(p) for p in neg_prototypes]),
                "neg_std_norm": np.std([np.linalg.norm(p) for p in neg_prototypes]),
            },
        }

        logger.info(
            f"Positive prototype consistency: {analysis['pos_prototype_consistency']['mean_similarity']:.3f} ± {analysis['pos_prototype_consistency']['std_similarity']:.3f}"
        )
        logger.info(
            f"Negative prototype consistency: {analysis['neg_prototype_consistency']['mean_similarity']:.3f} ± {analysis['neg_prototype_consistency']['std_similarity']:.3f}"
        )
        logger.info(
            f"Symmetry error: {analysis['symmetry_analysis']['mean_symmetry_error']:.6f} ± {analysis['symmetry_analysis']['std_symmetry_error']:.6f}"
        )

        return analysis

    def analyze_embedding_space_alignment(self, projection: Dict) -> Dict:
        """
        Analyze embedding space alignment using various metrics.

        This method computes alignment metrics that measure the quality and structure
        of the embedding space, including:
        - SAD (Sample Alignment Distance): Distance between samples across batches
        - SAA (Sample Alignment Accuracy): Accuracy of nearest neighbor matching
        - CAD (Class Alignment Distance): Class-specific alignment distances
        - CAC (Class Alignment Consistency): Consistency of class assignments
        - GPU (Gaussian Potential Uniformity): Uniformity using Gaussian potential
        - PEU (Probabilistic Entropy Uniformity): Uniformity using entropy

        Args:
            projection: Dictionary containing embeddings and labels from generate_embeddings_with_prototypes

        Returns:
            Dictionary containing all alignment metrics
        """
        logger.info("Analyzing embedding space alignment metrics...")

        # Get multi-view embeddings (required for contrastive learning analysis)
        multiview_embeddings = projection[
            "embeddings_nd"
        ]  # [N, num_variants, feat_dim]
        labels = projection["labels"]

        if multiview_embeddings.shape[0] != len(labels):
            logger.error("Embeddings and labels shape mismatch")
            return {}

        logger.info(
            f"Computing alignment metrics for {multiview_embeddings.shape[0]} samples with "
            f"{multiview_embeddings.shape[1]} variants and {multiview_embeddings.shape[2]} dimensions"
        )

        try:
            if multiview_embeddings.shape[1] < 2:
                single_view_embeddings = multiview_embeddings[:, 0, :]
                norms = np.linalg.norm(single_view_embeddings, axis=1, keepdims=True)
                alignment_input = single_view_embeddings / (norms + 1e-8)
                logger.info(
                    "Single-view prototype analysis detected; computing alignment metrics without variant-dependent SAD/SAA"
                )
            else:
                normalized_multiview_embeddings = np.zeros_like(multiview_embeddings)

                for i in range(multiview_embeddings.shape[1]):
                    variant_embeddings = multiview_embeddings[:, i, :]
                    norms = np.linalg.norm(variant_embeddings, axis=1, keepdims=True)
                    normalized_multiview_embeddings[:, i, :] = variant_embeddings / (
                        norms + 1e-8
                    )

                logger.info(
                    "Applied L2 normalization to each view of multi-view embeddings for alignment metrics"
                )
                alignment_input = normalized_multiview_embeddings

            alignment_metrics = calculate_all_alignment_metrics(
                embeddings=alignment_input,
                labels=labels,
            )

            # Log key metrics
            if "gpu" in alignment_metrics and "error" not in alignment_metrics["gpu"]:
                logger.info(
                    f"GPU - Class 0: {alignment_metrics['gpu']['class_0']:.3f}, Class 1: {alignment_metrics['gpu']['class_1']:.3f}"
                )

            if "peu" in alignment_metrics and "error" not in alignment_metrics["peu"]:
                logger.info(
                    f"PEU - Class 0: {alignment_metrics['peu']['class_0_mean']:.2f}%, Class 1: {alignment_metrics['peu']['class_1_mean']:.2f}%"
                )

            if "cad" in alignment_metrics and "error" not in alignment_metrics["cad"]:
                logger.info(
                    f"CAD - Class 0: {alignment_metrics['cad']['class_0']['mean']:.3f}±{alignment_metrics['cad']['class_0']['std']:.3f}"
                )
                logger.info(
                    f"CAD - Class 1: {alignment_metrics['cad']['class_1']['mean']:.3f}±{alignment_metrics['cad']['class_1']['std']:.3f}"
                )

            if "cac" in alignment_metrics and "error" not in alignment_metrics["cac"]:
                logger.info(
                    f"CAC - Class 0: {alignment_metrics['cac']['class_0']['mean']:.2f}±{alignment_metrics['cac']['class_0']['std']:.2f}%"
                )
                logger.info(
                    f"CAC - Class 1: {alignment_metrics['cac']['class_1']['mean']:.2f}±{alignment_metrics['cac']['class_1']['std']:.2f}%"
                )

            if "sad" in alignment_metrics and "error" not in alignment_metrics["sad"]:
                logger.info(
                    f"SAD - Class 0: {alignment_metrics['sad']['class_0']['mean']:.3f}±{alignment_metrics['sad']['class_0']['std']:.3f}"
                )
                logger.info(
                    f"SAD - Class 1: {alignment_metrics['sad']['class_1']['mean']:.3f}±{alignment_metrics['sad']['class_1']['std']:.3f}"
                )

            if "saa" in alignment_metrics and "error" not in alignment_metrics["saa"]:
                logger.info(
                    f"SAA - Class 0: {alignment_metrics['saa']['class_0']:.2f}%, Class 1: {alignment_metrics['saa']['class_1']:.2f}%"
                )

            logger.info("Embedding space alignment analysis complete")
            return alignment_metrics

        except Exception as e:
            logger.error(f"Error computing alignment metrics: {e}")
            return {"error": str(e)}

    def analyze_cosine_similarities(self, projection: Dict) -> Dict:
        """
        Analyze cosine similarities between samples in the embedding space.

        This method computes intra-class and inter-class similarities to understand
        how well the prototype ranking model separates different classes.

        Args:
            projection: Dictionary containing embeddings and labels

        Returns:
            Dictionary with cosine similarity analysis results
        """
        logger.info("Analyzing cosine similarities between samples...")

        # Use averaged embeddings for similarity analysis
        multiview_embeddings = projection[
            "embeddings_nd"
        ]  # [N, num_variants, feat_dim]
        averaged_embeddings = multiview_embeddings.mean(axis=1)  # [N, feat_dim]
        labels = projection["labels"]

        # Separate by class
        pos_mask = labels == 1
        neg_mask = labels == 0

        # Calculate cosine similarities between samples
        pos_embeddings = averaged_embeddings[pos_mask]
        neg_embeddings = averaged_embeddings[neg_mask]

        # Intra-class similarities
        pos_intra_sim = []
        neg_intra_sim = []

        if len(pos_embeddings) > 1:
            pos_sim_matrix = cosine_similarity(pos_embeddings)
            # Get upper triangular (excluding diagonal)
            pos_intra_sim = pos_sim_matrix[np.triu_indices(len(pos_embeddings), k=1)]

        if len(neg_embeddings) > 1:
            neg_sim_matrix = cosine_similarity(neg_embeddings)
            # Get upper triangular (excluding diagonal)
            neg_intra_sim = neg_sim_matrix[np.triu_indices(len(neg_embeddings), k=1)]

        # Inter-class similarities
        inter_sim = []
        if len(pos_embeddings) > 0 and len(neg_embeddings) > 0:
            inter_sim = cosine_similarity(pos_embeddings, neg_embeddings).flatten()

        # Calculate statistics
        analysis = {
            "cosine_similarities": {
                "pos_intra_mean": (
                    float(np.mean(pos_intra_sim)) if len(pos_intra_sim) > 0 else 0.0
                ),
                "pos_intra_std": (
                    float(np.std(pos_intra_sim)) if len(pos_intra_sim) > 0 else 0.0
                ),
                "neg_intra_mean": (
                    float(np.mean(neg_intra_sim)) if len(neg_intra_sim) > 0 else 0.0
                ),
                "neg_intra_std": (
                    float(np.std(neg_intra_sim)) if len(neg_intra_sim) > 0 else 0.0
                ),
                "inter_mean": float(np.mean(inter_sim)) if len(inter_sim) > 0 else 0.0,
                "inter_std": float(np.std(inter_sim)) if len(inter_sim) > 0 else 0.0,
            },
            "class_separation": {
                "pos_cluster_coherence": (
                    float(np.mean(pos_intra_sim)) if len(pos_intra_sim) > 0 else 0.0
                ),
                "neg_cluster_coherence": (
                    float(np.mean(neg_intra_sim)) if len(neg_intra_sim) > 0 else 0.0
                ),
                "inter_class_distance": (
                    1.0 - float(np.mean(inter_sim)) if len(inter_sim) > 0 else 0.0
                ),
            },
        }

        # Log key insights
        if len(pos_intra_sim) > 0:
            logger.info(
                f"Positive class coherence: {analysis['cosine_similarities']['pos_intra_mean']:.3f} ± {analysis['cosine_similarities']['pos_intra_std']:.3f}"
            )
        if len(neg_intra_sim) > 0:
            logger.info(
                f"Negative class coherence: {analysis['cosine_similarities']['neg_intra_mean']:.3f} ± {analysis['cosine_similarities']['neg_intra_std']:.3f}"
            )
        if len(inter_sim) > 0:
            logger.info(
                f"Inter-class separation: {analysis['class_separation']['inter_class_distance']:.3f}"
            )

        return analysis

    def analyze_threshold_performance(self) -> Dict:
        """
        Analyze threshold performance using out-of-fold predictions.

        This method uses the actual model predictions from cross-validation
        to evaluate different threshold optimization strategies, providing
        a more accurate assessment of model performance.

        Returns:
            Dictionary with threshold analysis results
        """
        logger.info("Analyzing threshold performance using out-of-fold predictions...")

        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Combine predictions and labels from all folds
        all_predictions = []
        all_labels = []

        for fold_num in sorted(self.predictions.keys()):
            preds = self.predictions[fold_num]
            labels = self.labels[fold_num].flatten()
            all_predictions.append(preds)
            all_labels.append(labels)

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        logger.info(f"Combined OOF data: {len(predictions)} samples")
        logger.info(
            f"Positive samples: {np.sum(labels == 1)} ({np.mean(labels == 1):.1%})"
        )
        logger.info(
            f"Negative samples: {np.sum(labels == 0)} ({np.mean(labels == 0):.1%})"
        )

        # Calculate overall metrics
        overall_metrics = {}
        if len(np.unique(labels)) > 1:
            try:
                roc_auc = roc_auc_score(labels, predictions)
                auprc = average_precision_score(labels, predictions)

                overall_metrics = {"roc_auc": roc_auc, "auprc": auprc}

                logger.info(f"Overall ROC-AUC: {roc_auc:.3f}")
                logger.info(f"Overall AUPRC: {auprc:.3f}")

            except Exception as e:
                logger.warning(f"Failed to compute overall metrics: {e}")

        # Test different threshold optimization methods
        threshold_methods = ["youden", "f1", "mcc"]
        threshold_results = {}

        for method in threshold_methods:
            try:
                optimal_threshold = find_optimal_threshold(
                    predictions=predictions, labels=labels, method=method
                )

                # Calculate metrics at this threshold
                y_pred = (predictions >= optimal_threshold).astype(int)
                precision = precision_score(labels, y_pred, zero_division=0)
                recall = recall_score(labels, y_pred, zero_division=0)
                f1 = f1_score(labels, y_pred, zero_division=0)
                mcc = matthews_corrcoef(labels, y_pred)

                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

                threshold_results[method] = {
                    "optimal_threshold": optimal_threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }

                logger.info(
                    f"{method} - Threshold: {optimal_threshold:.3f}, "
                    f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
                )

            except Exception as e:
                logger.warning(f"Failed to compute {method} threshold: {e}")

        # Find best performing method
        best_f1 = 0
        best_method = None
        best_metrics = None

        for method, results in threshold_results.items():
            if results["f1"] > best_f1:
                best_f1 = results["f1"]
                best_method = method
                best_metrics = results

        if best_method:
            logger.info(f"Best performing method: {best_method} with F1: {best_f1:.3f}")

        # Create comprehensive analysis
        analysis = {
            "overall_metrics": overall_metrics,
            "threshold_results": threshold_results,
            "best_method": best_method,
            "best_metrics": best_metrics,
            "data_summary": {
                "total_samples": len(predictions),
                "positive_samples": int(np.sum(labels == 1)),
                "negative_samples": int(np.sum(labels == 0)),
            },
        }

        return analysis

    def plot_threshold_analysis(self, threshold_analysis: Dict, save_dir: Path):
        """
        Plot threshold analysis results using the comprehensive visualization function.

        Args:
            threshold_analysis: Results from analyze_threshold_performance
            save_dir: Directory to save plots
        """
        logger.info("Creating comprehensive threshold analysis plots...")
        # Get predictions and labels from the analysis
        best_method = threshold_analysis.get("best_method")
        best_metrics = threshold_analysis.get("best_metrics")

        # Reconstruct predictions and labels from the analysis
        # We need to get the original predictions and labels from the OOF data
        all_predictions = []
        all_labels = []

        for fold_num in sorted(self.predictions.keys()):
            preds = self.predictions[fold_num]
            labels = self.labels[fold_num].flatten()
            all_predictions.append(preds)
            all_labels.append(labels)

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        # Use the comprehensive threshold analysis function
        plot_threshold_analysis_comprehensive(
            outputs=predictions,
            labels=labels,
            save_dir=save_dir,
            fold_number=None,  # Not fold-specific
            threshold_methods=["f1", "mcc", "youden"],
            optimal_threshold=(
                best_metrics["optimal_threshold"] if best_metrics else None
            ),
            threshold_method_used=best_method,
            logger=logger,
        )

        # Also create our custom summary plots for additional insights
        self._plot_threshold_summary(threshold_analysis, save_dir)

    def _plot_threshold_summary(self, threshold_analysis: Dict, save_dir: Path):
        """
        Create additional summary plots for threshold analysis.

        Args:
            threshold_analysis: Results from analyze_threshold_performance
            save_dir: Directory to save plots
        """
        logger.info("Creating threshold summary plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        threshold_results = threshold_analysis["threshold_results"]
        overall_metrics = threshold_analysis.get("overall_metrics", {})

        # 1. F1 Score Comparison
        methods = list(threshold_results.keys())
        f1_scores = [threshold_results[method]["f1"] for method in methods]

        bars = axes[0, 0].bar(methods, f1_scores, alpha=0.7)
        axes[0, 0].set_title("F1 Score by Threshold Method")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. Precision vs Recall
        precisions = [threshold_results[method]["precision"] for method in methods]
        recalls = [threshold_results[method]["recall"] for method in methods]

        scatter = axes[0, 1].scatter(recalls, precisions, s=100, alpha=0.7)
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision vs Recall")
        axes[0, 1].grid(True, alpha=0.3)

        # Add method labels
        for i, method in enumerate(methods):
            axes[0, 1].annotate(
                method,
                (recalls[i], precisions[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # 3. Optimal Thresholds
        thresholds = [
            threshold_results[method]["optimal_threshold"] for method in methods
        ]

        bars = axes[1, 0].bar(methods, thresholds, alpha=0.7, color="orange")
        axes[1, 0].set_title("Optimal Thresholds")
        axes[1, 0].set_ylabel("Threshold")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, thresholds):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 4. Overall Metrics
        if overall_metrics:
            metrics_names = list(overall_metrics.keys())
            metrics_values = list(overall_metrics.values())

            bars = axes[1, 1].bar(
                metrics_names, metrics_values, alpha=0.7, color="green"
            )
            axes[1, 1].set_title("Overall Model Performance")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No overall metrics available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Overall Model Performance")

        plt.tight_layout()
        plt.savefig(save_dir / "threshold_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Threshold summary plots saved to {save_dir / 'threshold_summary.png'}"
        )

    def plot_distance_analysis(
        self,
        projection: Dict,
        distance_analysis: Dict,
        save_dir: Path,
    ):
        """Plot distance analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        embeddings_2d = projection["embeddings_2d"]
        labels = projection["labels"]
        prototypes_2d = projection["prototypes_2d"]

        # Calculate 2D distances to learned prototypes for visualization
        pos_proto_2d = prototypes_2d[self.POS_IDX]
        neg_proto_2d = prototypes_2d[self.NEG_IDX]

        pos_distances = np.linalg.norm(embeddings_2d - pos_proto_2d, axis=1)
        neg_distances = np.linalg.norm(embeddings_2d - neg_proto_2d, axis=1)

        # 1. Distance distribution by class (2D visualization)
        pos_mask = labels == 1
        neg_mask = labels == 0

        axes[0, 0].hist(
            pos_distances[pos_mask],
            bins=30,
            alpha=0.7,
            label="Positive samples",
            density=True,
            color="green",
        )
        axes[0, 0].hist(
            neg_distances[neg_mask],
            bins=30,
            alpha=0.7,
            label="Negative samples",
            density=True,
            color="red",
        )
        axes[0, 0].set_title("Distance to Respective Prototype (2D Visualization)")
        axes[0, 0].set_xlabel("Distance")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Distance comparison (2D visualization)
        axes[0, 1].scatter(
            pos_distances[pos_mask],
            neg_distances[pos_mask],
            alpha=0.6,
            color="green",
            label="Positive samples",
            s=20,
        )
        axes[0, 1].scatter(
            pos_distances[neg_mask],
            neg_distances[neg_mask],
            alpha=0.6,
            color="red",
            label="Negative samples",
            s=20,
        )
        axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
        axes[0, 1].set_title("Distance to Positive vs Negative Prototype (2D)")
        axes[0, 1].set_xlabel("Distance to Positive Prototype")
        axes[0, 1].set_ylabel("Distance to Negative Prototype")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Similarity analysis (from 1024D analysis)
        # Extract similarity values from the analysis
        pos_sim_pos = distance_analysis["pos_samples_to_pos_proto"]["mean"]
        pos_sim_neg = distance_analysis["pos_samples_to_neg_proto"]["mean"]
        neg_sim_pos = distance_analysis["neg_samples_to_pos_proto"]["mean"]
        neg_sim_neg = distance_analysis["neg_samples_to_neg_proto"]["mean"]

        # Create similarity comparison plot
        categories = ["Pos→Pos", "Pos→Neg", "Neg→Pos", "Neg→Neg"]
        similarities = [pos_sim_pos, pos_sim_neg, neg_sim_pos, neg_sim_neg]
        colors = ["green", "lightgreen", "lightcoral", "red"]

        bars = axes[1, 0].bar(categories, similarities, color=colors, alpha=0.7)
        axes[1, 0].set_title("Cosine Similarity to Prototypes (1024D Analysis)")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, similarities):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 4. Prototype effectiveness summary
        effectiveness_metrics = [
            distance_analysis["pos_closer_to_pos"],
            distance_analysis["neg_closer_to_neg"],
            distance_analysis["prototype_separation_2d"],
        ]
        metric_names = ["Pos→Pos", "Neg→Neg", "Separation (2D)"]

        bars = axes[1, 1].bar(metric_names, effectiveness_metrics, alpha=0.7)
        bars[0].set_color("green")
        bars[1].set_color("red")
        bars[2].set_color("blue")
        axes[1, 1].set_title("Prototype Effectiveness Metrics")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, effectiveness_metrics):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_dir / "distance_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved distance analysis to {save_dir / 'distance_analysis.png'}")

    def plot_cosine_similarity_analysis(self, cosine_analysis: Dict, save_dir: Path):
        """Plot cosine similarity analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Cosine similarities comparison
        sim_stats = cosine_analysis["cosine_similarities"]
        sim_categories = ["Pos Intra", "Neg Intra", "Inter"]
        sim_means = [
            sim_stats["pos_intra_mean"],
            sim_stats["neg_intra_mean"],
            sim_stats["inter_mean"],
        ]
        sim_stds = [
            sim_stats["pos_intra_std"],
            sim_stats["neg_intra_std"],
            sim_stats["inter_std"],
        ]
        sim_colors = ["green", "red", "blue"]

        bars = axes[0, 0].bar(
            sim_categories,
            sim_means,
            yerr=sim_stds,
            capsize=5,
            color=sim_colors,
            alpha=0.7,
        )
        axes[0, 0].set_title("Cosine Similarities")
        axes[0, 0].set_ylabel("Similarity")
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, sim_means, sim_stds):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. Class separation metrics
        sep_stats = cosine_analysis["class_separation"]
        sep_categories = ["Pos Coherence", "Neg Coherence", "Inter Distance"]
        sep_values = [
            sep_stats["pos_cluster_coherence"],
            sep_stats["neg_cluster_coherence"],
            sep_stats["inter_class_distance"],
        ]
        sep_colors = ["green", "red", "blue"]

        bars = axes[0, 1].bar(sep_categories, sep_values, color=sep_colors, alpha=0.7)
        axes[0, 1].set_title("Class Separation Metrics")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, sep_values):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 3. Intra-class vs Inter-class comparison
        intra_means = [sim_stats["pos_intra_mean"], sim_stats["neg_intra_mean"]]
        intra_stds = [sim_stats["pos_intra_std"], sim_stats["neg_intra_std"]]
        intra_labels = ["Positive", "Negative"]
        intra_colors = ["green", "red"]

        bars = axes[1, 0].bar(
            intra_labels,
            intra_means,
            yerr=intra_stds,
            capsize=5,
            color=intra_colors,
            alpha=0.7,
        )
        axes[1, 0].set_title("Intra-class Similarities")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, intra_means, intra_stds):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
            )

        # 4. Summary statistics
        axes[1, 1].axis("off")
        stats_text = f"""
        Cosine Similarity Analysis
        
        Positive intra-class: {sim_stats["pos_intra_mean"]:.3f} ± {sim_stats["pos_intra_std"]:.3f}
        Negative intra-class: {sim_stats["neg_intra_mean"]:.3f} ± {sim_stats["neg_intra_std"]:.3f}
        Inter-class: {sim_stats["inter_mean"]:.3f} ± {sim_stats["inter_std"]:.3f}
        
        Class Separation:
        - Positive coherence: {sep_stats["pos_cluster_coherence"]:.3f}
        - Negative coherence: {sep_stats["neg_cluster_coherence"]:.3f}
        - Inter-class distance: {sep_stats["inter_class_distance"]:.3f}
        """
        axes[1, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(
            save_dir / "cosine_similarity_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Saved cosine similarity analysis to {save_dir / 'cosine_similarity_analysis.png'}"
        )

    def plot_prototype_consistency(self, prototype_consistency: Dict, save_dir: Path):
        """Plot prototype consistency analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Positive prototype consistency
        pos_stats = prototype_consistency["pos_prototype_consistency"]
        axes[0, 0].bar(
            ["Mean", "Std", "Min", "Max"],
            [
                pos_stats["mean_similarity"],
                pos_stats["std_similarity"],
                pos_stats["min_similarity"],
                pos_stats["max_similarity"],
            ],
            color="green",
            alpha=0.7,
        )
        axes[0, 0].set_title("Positive Prototype Consistency")
        axes[0, 0].set_ylabel("Cosine Similarity")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Negative prototype consistency
        neg_stats = prototype_consistency["neg_prototype_consistency"]
        axes[0, 1].bar(
            ["Mean", "Std", "Min", "Max"],
            [
                neg_stats["mean_similarity"],
                neg_stats["std_similarity"],
                neg_stats["min_similarity"],
                neg_stats["max_similarity"],
            ],
            color="red",
            alpha=0.7,
        )
        axes[0, 1].set_title("Negative Prototype Consistency")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Symmetry analysis
        sym_stats = prototype_consistency["symmetry_analysis"]
        axes[1, 0].bar(
            ["Mean Error", "Std Error", "Max Error"],
            [
                sym_stats["mean_symmetry_error"],
                sym_stats["std_symmetry_error"],
                sym_stats["max_symmetry_error"],
            ],
            color="blue",
            alpha=0.7,
        )
        axes[1, 0].set_title("Symmetry Error (p + (-p) should be 0)")
        axes[1, 0].set_ylabel("L2 Norm")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Prototype norms
        norm_stats = prototype_consistency["prototype_norms"]
        categories = ["Pos Mean", "Pos Std", "Neg Mean", "Neg Std"]
        values = [
            norm_stats["pos_mean_norm"],
            norm_stats["pos_std_norm"],
            norm_stats["neg_mean_norm"],
            norm_stats["neg_std_norm"],
        ]
        colors = ["green", "lightgreen", "red", "lightcoral"]

        bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
        axes[1, 1].set_title("Prototype Norms")
        axes[1, 1].set_ylabel("L2 Norm")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            save_dir / "prototype_consistency_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Saved prototype consistency analysis to {save_dir / 'prototype_consistency_analysis.png'}"
        )

    def plot_alignment_metrics(self, alignment_metrics: Dict, save_dir: Path):
        """
        Plot alignment metrics analysis.

        Args:
            alignment_metrics: Results from analyze_embedding_space_alignment
            save_dir: Directory to save plots
        """
        logger.info("Creating alignment metrics visualization...")

        # Filter out metrics with errors
        valid_metrics = {k: v for k, v in alignment_metrics.items() if "error" not in v}

        if not valid_metrics:
            logger.warning("No valid alignment metrics to plot")
            return

        # Determine number of subplots needed
        n_metrics = len(valid_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle different subplot configurations
        if n_metrics == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            # Ensure axes is a numpy array
            axes = np.array(axes)

        # Now axes should always be a numpy array, so flatten() will work
        axes = axes.flatten()

        metric_idx = 0

        # Plot GPU (Gaussian Potential Uniformity)
        if "gpu" in valid_metrics:
            gpu_data = valid_metrics["gpu"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            values = [gpu_data["class_0"], gpu_data["class_1"]]
            colors = ["red", "green"]

            bars = ax.bar(classes, values, color=colors, alpha=0.7)
            ax.set_title("Gaussian Potential Uniformity (GPU)")
            ax.set_ylabel("Uniformity Score")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            metric_idx += 1

        # Plot PEU (Probabilistic Entropy Uniformity)
        if "peu" in valid_metrics:
            peu_data = valid_metrics["peu"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            values = [peu_data["class_0_mean"], peu_data["class_1_mean"]]
            colors = ["red", "green"]

            bars = ax.bar(classes, values, color=colors, alpha=0.7)
            ax.set_title("Probabilistic Entropy Uniformity (PEU)")
            ax.set_ylabel("Uniformity (%)")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                )

            metric_idx += 1

        # Plot CAD (Class Alignment Distance)
        if "cad" in valid_metrics:
            cad_data = valid_metrics["cad"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            means = [cad_data["class_0"]["mean"], cad_data["class_1"]["mean"]]
            stds = [cad_data["class_0"]["std"], cad_data["class_1"]["std"]]
            colors = ["red", "green"]

            # For CAD, std is always 0.0 since it's a single value per class
            if stds[0] == 0.0 and stds[1] == 0.0:
                bars = ax.bar(classes, means, color=colors, alpha=0.7)
                ax.set_title("Class Alignment Distance (CAD)")
                ax.set_ylabel("Average Distance")
            else:
                bars = ax.bar(
                    classes, means, yerr=stds, capsize=5, color=colors, alpha=0.7
                )
                ax.set_title("Class Alignment Distance (CAD)")
                ax.set_ylabel("Distance")

            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                if std == 0.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{mean:.3f}±{std:.3f}",
                        ha="center",
                        va="bottom",
                    )

            metric_idx += 1

        # Plot CAC (Class Alignment Consistency)
        if "cac" in valid_metrics:
            cac_data = valid_metrics["cac"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            means = [cac_data["class_0"]["mean"], cac_data["class_1"]["mean"]]
            stds = [cac_data["class_0"]["std"], cac_data["class_1"]["std"]]
            colors = ["red", "green"]

            bars = ax.bar(classes, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.set_title("Class Alignment Consistency (CAC)")
            ax.set_ylabel("Consistency (%)")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.1f}±{std:.1f}%",
                    ha="center",
                    va="bottom",
                )

            metric_idx += 1

        # Plot SAD (Sample Alignment Distance)
        if "sad" in valid_metrics:
            sad_data = valid_metrics["sad"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            means = [sad_data["class_0"]["mean"], sad_data["class_1"]["mean"]]
            stds = [sad_data["class_0"]["std"], sad_data["class_1"]["std"]]
            colors = ["red", "green"]

            bars = ax.bar(classes, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.set_title("Sample Alignment Distance (SAD)")
            ax.set_ylabel("Distance")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.3f}±{std:.3f}",
                    ha="center",
                    va="bottom",
                )

            metric_idx += 1

        # Plot SAA (Sample Alignment Accuracy)
        if "saa" in valid_metrics:
            saa_data = valid_metrics["saa"]
            ax = axes[metric_idx]

            classes = ["Class 0", "Class 1"]
            values = [saa_data["class_0"], saa_data["class_1"]]
            colors = ["red", "green"]

            bars = ax.bar(classes, values, color=colors, alpha=0.7)
            ax.set_title("Sample Alignment Accuracy (SAA)")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                )

            metric_idx += 1

        # Hide unused subplots
        for i in range(metric_idx, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        save_path = save_dir / "alignment_metrics_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Alignment metrics visualization saved to {save_path}")

    def analyze_distance_score_confidence(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """Analyze distance scores with confidence intervals."""
        logger.info("Analyzing distance scores with confidence intervals...")

        # Convert to distance scores
        distance_scores = np.log(predictions / (1 - predictions + 1e-8))

        # Separate by class
        pos_scores = distance_scores[labels == 1]
        neg_scores = distance_scores[labels == 0]

        # Calculate confidence intervals
        def confidence_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)
            h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
            return mean, mean - h, mean + h

        pos_mean, pos_lower, pos_upper = confidence_interval(pos_scores)
        neg_mean, neg_lower, neg_upper = confidence_interval(neg_scores)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(pos_scores) - 1) * np.var(pos_scores)
                + (len(neg_scores) - 1) * np.var(neg_scores)
            )
            / (len(pos_scores) + len(neg_scores) - 2)
        )
        cohens_d = (pos_mean - neg_mean) / pooled_std

        analysis = {
            "pos_scores": {
                "mean": pos_mean,
                "std": np.std(pos_scores),
                "ci_lower": pos_lower,
                "ci_upper": pos_upper,
                "n": len(pos_scores),
            },
            "neg_scores": {
                "mean": neg_mean,
                "std": np.std(neg_scores),
                "ci_lower": neg_lower,
                "ci_upper": neg_upper,
                "n": len(neg_scores),
            },
            "separation": {
                "mean": pos_mean - neg_mean,
                "cohens_d": cohens_d,
                "effect_size": (
                    "large"
                    if abs(cohens_d) > 0.8
                    else "medium"
                    if abs(cohens_d) > 0.5
                    else "small"
                ),
            },
        }

        logger.info(
            f"Positive scores: {pos_mean:.3f} [{pos_lower:.3f}, {pos_upper:.3f}]"
        )
        logger.info(
            f"Negative scores: {neg_mean:.3f} [{neg_lower:.3f}, {neg_upper:.3f}]"
        )
        logger.info(
            f"Separation: {analysis['separation']['mean']:.3f} (Cohen's d: {cohens_d:.3f})"
        )

        return analysis

    def plot_distance_score_confidence(self, distance_analysis: Dict, save_dir: Path):
        """Plot distance score analysis with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confidence interval plot
        pos_stats = distance_analysis["pos_scores"]
        neg_stats = distance_analysis["neg_scores"]

        classes = ["Positive", "Negative"]
        means = [pos_stats["mean"], neg_stats["mean"]]
        ci_lower = [pos_stats["ci_lower"], neg_stats["ci_lower"]]
        ci_upper = [pos_stats["ci_upper"], neg_stats["ci_upper"]]

        yerr = np.array(
            [[means[i] - ci_lower[i], ci_upper[i] - means[i]] for i in range(2)]
        ).T

        bars = axes[0, 0].bar(classes, means, yerr=yerr, capsize=5, alpha=0.7)
        bars[0].set_color("green")
        bars[1].set_color("red")
        axes[0, 0].set_title("distance score Means with 95% CI")
        axes[0, 0].set_ylabel("distance score (logits)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Effect size visualization
        cohens_d = distance_analysis["separation"]["cohens_d"]
        effect_size = distance_analysis["separation"]["effect_size"]

        # Create effect size visualization
        x = np.linspace(-3, 3, 1000)
        pos_dist = stats.norm.pdf(x, pos_stats["mean"], pos_stats["std"])
        neg_dist = stats.norm.pdf(x, neg_stats["mean"], neg_stats["std"])

        axes[0, 1].plot(x, pos_dist, color="green", label="Positive", linewidth=2)
        axes[0, 1].plot(x, neg_dist, color="red", label="Negative", linewidth=2)
        axes[0, 1].fill_between(x, pos_dist, alpha=0.3, color="green")
        axes[0, 1].fill_between(x, neg_dist, alpha=0.3, color="red")
        axes[0, 1].set_title(f"Score Distributions (Cohen's d: {cohens_d:.3f})")
        axes[0, 1].set_xlabel("distance score (logits)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Separation analysis
        separation = distance_analysis["separation"]["mean"]
        axes[1, 0].bar(["Score Separation"], [separation], color="blue", alpha=0.7)
        axes[1, 0].set_title("Score Separation")
        axes[1, 0].set_ylabel("Separation (logits)")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value label
        axes[1, 0].text(0, separation, f"{separation:.3f}", ha="center", va="bottom")

        # 4. Effect size interpretation
        effect_sizes = ["Small", "Medium", "Large"]
        cohens_d_ranges = [0.2, 0.5, 0.8]
        colors = ["lightblue", "orange", "red"]

        for i, (effect, threshold, color) in enumerate(
            zip(effect_sizes, cohens_d_ranges, colors)
        ):
            axes[1, 1].barh([effect], [threshold], color=color, alpha=0.7)

        # Mark current effect size
        current_effect_idx = (
            0 if abs(cohens_d) <= 0.2 else 1 if abs(cohens_d) <= 0.5 else 2
        )
        axes[1, 1].barh(
            [effect_sizes[current_effect_idx]],
            [abs(cohens_d)],
            color="darkblue",
            alpha=0.9,
            label=f"Current: {abs(cohens_d):.3f}",
        )
        axes[1, 1].set_title("Effect Size Interpretation")
        axes[1, 1].set_xlabel("|Cohen's d|")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / "distance_score_confidence.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Saved distance score confidence analysis to {save_dir / 'distance_score_confidence.png'}"
        )

    def run_analysis(
        self,
        output_dir: Path = None,
        fold: int = 1,
        sample_size: int = 100,
        is_single_stage: bool = False,
    ):
        """Run analysis with prototypes from checkpoints.

        Supports the retained prototype contrastive training path.

        Args:
            output_dir: Directory to save analysis results
            fold: Fold number to use for model loading and inference
            sample_size: Number of samples to use for visualization
            is_single_stage: True for single-stage training (fold_X/checkpoint.pt),
                         False for two-stage (fold_X/finetuning/checkpoint.pt)
        """
        if output_dir is None:
            output_dir = self.run_dir / "prototype_analysis"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting analysis with prototypes (fold {fold}, sample_size {sample_size}), saving to {output_dir}"
        )
        logger.info(
            f"Multi-view analysis enabled: Will use all available views for proper contrastive learning analysis"
        )

        # Load prototypes from checkpoints
        prototypes_by_fold = self.load_prototypes_from_checkpoints(
            is_single_stage=is_single_stage
        )

        if not prototypes_by_fold:
            logger.error("No prototypes could be loaded from checkpoints")
            return {}

        # Combine data from all folds
        all_predictions = []
        all_labels = []

        for fold_num in sorted(self.predictions.keys()):
            preds = self.predictions[fold_num]
            labels = self.labels[fold_num].flatten()
            all_predictions.append(preds)
            all_labels.append(labels)

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        logger.info(f"Combined data: {len(predictions)} samples")
        logger.info(f"Positive samples: {np.sum(labels == 1)}")
        logger.info(f"Negative samples: {np.sum(labels == 0)})")

        # 1. generate embeddings and 2D projections
        projection = self.generate_embeddings_with_prototypes(
            prototypes_by_fold,
            fold=fold,
            sample_size=sample_size,
            is_single_stage=is_single_stage,
        )

        # 2. Visualize embedding space with prototypes
        self.visualize_embedding_space_with_prototypes(projection, output_dir)

        # 3. Analyze prototype consistency across folds
        prototype_consistency = self.analyze_prototype_consistency(prototypes_by_fold)
        self.plot_prototype_consistency(prototype_consistency, output_dir)

        # 4. Analyze prototype distances using prototypes
        prototype_distance_analysis = self.analyze_prototype_distances(projection)
        self.plot_distance_analysis(projection, prototype_distance_analysis, output_dir)

        # 5. NEW: Analyze cosine similarities between samples
        cosine_similarity_analysis = self.analyze_cosine_similarities(projection)
        self.plot_cosine_similarity_analysis(cosine_similarity_analysis, output_dir)

        # 6. NEW: Analyze embedding space alignment metrics
        alignment_analysis = self.analyze_embedding_space_alignment(projection)
        self.plot_alignment_metrics(alignment_analysis, output_dir)

        # 7. Distance score analysis
        distance_score_analysis = self.analyze_distance_score_confidence(
            predictions, labels
        )
        self.plot_distance_score_confidence(distance_score_analysis, output_dir)

        # 8. NEW: Standalone threshold analysis using out-of-fold predictions
        threshold_analysis = self.analyze_threshold_performance()
        self.plot_threshold_analysis(threshold_analysis, output_dir)

        # 9. Save comprehensive results
        summary = {
            "embedding_analysis": {
                "n_samples": len(predictions),
                "n_positive": np.sum(labels == 1),
                "n_negative": np.sum(labels == 0),
            },
            # Compact projection summary for report
            "projection": {
                "pos_prototype_2d": projection["prototypes_2d"][self.POS_IDX].tolist(),
                "neg_prototype_2d": projection["prototypes_2d"][self.NEG_IDX].tolist(),
                "prototype_distance_2d": projection["prototype_distance_2d"],
            },
            "prototype_consistency": prototype_consistency,
            "prototype_distance_analysis": prototype_distance_analysis,
            "cosine_similarity_analysis": cosine_similarity_analysis,
            "alignment_analysis": alignment_analysis,  # NEW: Add alignment analysis
            "distance_score_analysis": distance_score_analysis,
            "threshold_analysis": threshold_analysis,  # NEW: Add threshold analysis
            "analysis_parameters": {
                "fold": fold,
                "sample_size": sample_size,
            },
        }

        import json

        with open(output_dir / "analysis_summary_with_prototypes.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 10. Create summary report
        self.create_summary_report_with_prototypes(summary, output_dir)

        logger.info(f"Prototype analysis complete! Results saved to {output_dir}")
        return summary

    def create_summary_report_with_prototypes(self, summary: Dict, output_dir: Path):
        """Create a text summary report with prototype analysis."""
        report = []
        report.append("=" * 80)
        report.append("PROTOTYPE ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Prototype consistency analysis
        report.append("PROTOTYPE CONSISTENCY ANALYSIS:")
        report.append("-" * 40)
        pc = summary["prototype_consistency"]
        pos_cons = pc["pos_prototype_consistency"]
        neg_cons = pc["neg_prototype_consistency"]
        sym_analysis = pc["symmetry_analysis"]

        report.append(
            f"Positive prototype consistency: {pos_cons['mean_similarity']:.3f} ± {pos_cons['std_similarity']:.3f}"
        )
        report.append(
            f"Negative prototype consistency: {neg_cons['mean_similarity']:.3f} ± {neg_cons['std_similarity']:.3f}"
        )
        report.append(
            f"Symmetry error: {sym_analysis['mean_symmetry_error']:.6f} ± {sym_analysis['std_symmetry_error']:.6f}"
        )
        report.append("")

        # Prototype locations (2D representation)
        report.append("PROTOTYPE LOCATIONS (2D Representation):")
        report.append("-" * 40)
        prj = summary["projection"]
        report.append(
            f"Positive prototype: [{prj['pos_prototype_2d'][0]:.3f}, {prj['pos_prototype_2d'][1]:.3f}]"
        )
        report.append(
            f"Negative prototype: [{prj['neg_prototype_2d'][0]:.3f}, {prj['neg_prototype_2d'][1]:.3f}]"
        )
        report.append(f"Prototype separation (2D): {prj['prototype_distance_2d']:.3f}")
        report.append("")

        # Distance analysis
        report.append("DISTANCE ANALYSIS:")
        report.append("-" * 40)
        da = summary["prototype_distance_analysis"]
        report.append(
            f"Positive samples closer to positive prototype: {da['pos_closer_to_pos']:.1%}"
        )
        report.append(
            f"Negative samples closer to negative prototype: {da['neg_closer_to_neg']:.1%}"
        )
        report.append(
            f"Ranking accuracy (correct direction): {da['ranking_accuracy']:.1%}"
        )
        report.append("")

        # NEW: Cosine similarity analysis
        if "cosine_similarity_analysis" in summary:
            report.append("COSINE SIMILARITY ANALYSIS:")
            report.append("-" * 40)
            csa = summary["cosine_similarity_analysis"]
            sims = csa["cosine_similarities"]
            sep = csa["class_separation"]

            report.append("Cosine Similarities:")
            report.append(
                f"  Positive intra-class: {sims['pos_intra_mean']:.3f} ± {sims['pos_intra_std']:.3f}"
            )
            report.append(
                f"  Negative intra-class: {sims['neg_intra_mean']:.3f} ± {sims['neg_intra_std']:.3f}"
            )
            report.append(
                f"  Inter-class: {sims['inter_mean']:.3f} ± {sims['inter_std']:.3f}"
            )
            report.append("")

            report.append("Class Separation:")
            report.append(f"  Positive coherence: {sep['pos_cluster_coherence']:.3f}")
            report.append(f"  Negative coherence: {sep['neg_cluster_coherence']:.3f}")
            report.append(f"  Inter-class distance: {sep['inter_class_distance']:.3f}")
            report.append("")

        # NEW: Embedding space alignment analysis
        if "alignment_analysis" in summary:
            report.append("EMBEDDING SPACE ALIGNMENT METRICS:")
            report.append("-" * 40)
            aa = summary["alignment_analysis"]

            # GPU (Gaussian Potential Uniformity)
            if "gpu" in aa and "error" not in aa["gpu"]:
                gpu = aa["gpu"]
                report.append("Gaussian Potential Uniformity (GPU):")
                report.append(f"  Class 0: {gpu['class_0']:.3f}")
                report.append(f"  Class 1: {gpu['class_1']:.3f}")
                report.append(f"  Overall: {gpu['mean']:.3f}")
                report.append("")

            # PEU (Probabilistic Entropy Uniformity)
            if "peu" in aa and "error" not in aa["peu"]:
                peu = aa["peu"]
                report.append("Probabilistic Entropy Uniformity (PEU):")
                report.append(f"  Class 0: {peu['class_0_mean']:.2f}%")
                report.append(f"  Class 1: {peu['class_1_mean']:.2f}%")
                report.append(f"  Overall: {peu['overall_mean']:.2f}%")
                report.append("")

            # CAD (Class Alignment Distance)
            if "cad" in aa and "error" not in aa["cad"]:
                cad = aa["cad"]
                report.append("Class Alignment Distance (CAD):")
                report.append(
                    f"  Class 0: {cad['class_0']['mean']:.3f} ± {cad['class_0']['std']:.3f}"
                )
                report.append(
                    f"  Class 1: {cad['class_1']['mean']:.3f} ± {cad['class_1']['std']:.3f}"
                )
                report.append("")

            # CAC (Class Alignment Consistency)
            if "cac" in aa and "error" not in aa["cac"]:
                cac = aa["cac"]
                report.append("Class Alignment Consistency (CAC):")
                report.append(
                    f"  Class 0: {cac['class_0']['mean']:.2f} ± {cac['class_0']['std']:.2f}%"
                )
                report.append(
                    f"  Class 1: {cac['class_1']['mean']:.2f} ± {cac['class_1']['std']:.2f}%"
                )
                report.append("")

            # SAD (Sample Alignment Distance) - if available
            if "sad" in aa and "error" not in aa["sad"]:
                sad = aa["sad"]
                report.append("Sample Alignment Distance (SAD):")
                report.append(
                    f"  Class 0: {sad['class_0']['mean']:.3f} ± {sad['class_0']['std']:.3f}"
                )
                report.append(
                    f"  Class 1: {sad['class_1']['mean']:.3f} ± {sad['class_1']['std']:.3f}"
                )
                report.append("")

            # SAA (Sample Alignment Accuracy) - if available
            if "saa" in aa and "error" not in aa["saa"]:
                saa = aa["saa"]
                report.append("Sample Alignment Accuracy (SAA):")
                report.append(f"  Class 0: {saa['class_0']:.2f}%")
                report.append(f"  Class 1: {saa['class_1']:.2f}%")
                report.append(f"  Overall: {saa['mean']:.2f}%")
                report.append("")

        # NEW: Standalone threshold analysis using out-of-fold predictions
        if "threshold_analysis" in summary:
            report.append("THRESHOLD ANALYSIS (Out-of-Fold Predictions):")
            report.append("-" * 40)
            ta = summary["threshold_analysis"]

            # Data summary
            if "data_summary" in ta:
                ds = ta["data_summary"]
                report.append(f"Total samples: {ds['total_samples']}")
                report.append(f"Positive samples: {ds['positive_samples']}")
                report.append(f"Negative samples: {ds['negative_samples']}")
                report.append("")

            # Overall metrics
            if "overall_metrics" in ta:
                om = ta["overall_metrics"]
                report.append("Overall Performance:")
                report.append(f"  ROC-AUC: {om['roc_auc']:.3f}")
                report.append(f"  AUPRC: {om['auprc']:.3f}")
                report.append("")

            # Threshold results
            if "threshold_results" in ta:
                report.append("Threshold Optimization Results:")
                tr = ta["threshold_results"]

                for method, results in tr.items():
                    report.append(f"  {method}:")
                    report.append(
                        f"    Optimal threshold: {results['optimal_threshold']:.3f}"
                    )
                    report.append(f"    Precision: {results['precision']:.3f}")
                    report.append(f"    Recall: {results['recall']:.3f}")
                    report.append(f"    F1 Score: {results['f1']:.3f}")
                    report.append(f"    MCC: {results['mcc']:.3f}")
                    report.append(
                        f"    Confusion Matrix: TP={results['tp']}, TN={results['tn']}, FP={results['fp']}, FN={results['fn']}"
                    )
                    report.append("")

        # distance score analysis
        report.append("DISTANCE SCORE ANALYSIS:")
        report.append("-" * 40)
        ra = summary["distance_score_analysis"]
        pos_stats = ra["pos_scores"]
        neg_stats = ra["neg_scores"]
        sep_stats = ra["separation"]

        report.append(
            f"Positive scores: {pos_stats['mean']:.3f} ± {pos_stats['std']:.3f}"
        )
        report.append(
            f"Negative scores: {neg_stats['mean']:.3f} ± {neg_stats['std']:.3f}"
        )
        report.append(f"Score separation: {sep_stats['mean']:.3f}")
        report.append(
            f"Effect size (Cohen's d): {sep_stats['cohens_d']:.3f} ({sep_stats['effect_size']})"
        )
        report.append("")

        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)

        # Prototype consistency
        if pos_cons["mean_similarity"] > 0.8:
            report.append("High consistency in positive prototypes across folds")
        else:
            report.append("Low consistency in positive prototypes across folds")

        if neg_cons["mean_similarity"] > 0.8:
            report.append("High consistency in negative prototypes across folds")
        else:
            report.append("Low consistency in negative prototypes across folds")

        # Prototype effectiveness
        effectiveness = (da["pos_closer_to_pos"] + da["neg_closer_to_neg"]) / 2
        if effectiveness > 0.8:
            report.append("✓ Excellent prototype separation")
        elif effectiveness > 0.7:
            report.append("✓ Good prototype separation")
        else:
            report.append("⚠ Room for improvement in prototype separation")

        # NEW: Cosine similarity insights
        if "cosine_similarity_analysis" in summary:
            csa = summary["cosine_similarity_analysis"]
            sims = csa["cosine_similarities"]
            sep = csa["class_separation"]

            # Positive class coherence
            if sep["pos_cluster_coherence"] > 0.7:
                report.append("✓ High positive class coherence")
            else:
                report.append("⚠ Room for improvement in positive class coherence")

            # Negative class coherence
            if sep["neg_cluster_coherence"] > 0.7:
                report.append("✓ High negative class coherence")
            else:
                report.append("⚠ Room for improvement in negative class coherence")

            # Inter-class separation
            if sep["inter_class_distance"] > 0.3:
                report.append("✓ Good inter-class separation")
            else:
                report.append("⚠ Room for improvement in inter-class separation")

        # Effect size
        if abs(sep_stats["cohens_d"]) > 0.8:
            report.append("✓ Large effect size in distance scores")
        elif abs(sep_stats["cohens_d"]) > 0.5:
            report.append("✓ Medium effect size in distance scores")
        else:
            report.append("⚠ Small effect size in distance scores")

        # Alignment metrics insights
        if "alignment_analysis" in summary:
            aa = summary["alignment_analysis"]

            # GPU insights
            if "gpu" in aa and "error" not in aa["gpu"]:
                gpu_mean = aa["gpu"]["mean"]
                if gpu_mean < 0.1:
                    report.append("✓ Low embedding space uniformity (GPU)")
                elif gpu_mean > 0.3:
                    report.append("⚠ High embedding space uniformity (GPU)")
                else:
                    report.append("⚠ Moderate embedding space uniformity (GPU)")

            # PEU insights
            if "peu" in aa and "error" not in aa["peu"]:
                peu_overall = aa["peu"]["overall_mean"]
                if peu_overall > 80:
                    report.append("✓ Excellent entropy uniformity (PEU)")
                elif peu_overall > 60:
                    report.append("✓ Good entropy uniformity (PEU)")
                else:
                    report.append("⚠ Room for improvement in entropy uniformity (PEU)")

            # CAC insights
            if "cac" in aa and "error" not in aa["cac"]:
                cac_0 = aa["cac"]["class_0"]["mean"]
                cac_1 = aa["cac"]["class_1"]["mean"]
                if cac_0 > 80 and cac_1 > 80:
                    report.append("✓ Excellent class alignment consistency")
                elif cac_0 > 60 and cac_1 > 60:
                    report.append("✓ Good class alignment consistency")
                else:
                    report.append(
                        "⚠ Room for improvement in class alignment consistency"
                    )

            # SAA insights (if available)
            if "saa" in aa and "error" not in aa["saa"]:
                saa_mean = aa["saa"]["mean"]
                if saa_mean > 80:
                    report.append("✓ Excellent sample alignment accuracy")
                elif saa_mean > 60:
                    report.append("✓ Good sample alignment accuracy")
                else:
                    report.append("⚠ Room for improvement in sample alignment accuracy")

        # Threshold analysis insights
        if "threshold_analysis" in summary:
            ta = summary["threshold_analysis"]
            if "best_method" in ta and ta["best_method"]:
                report.append(f"✓ Best threshold method: {ta['best_method']}")
                if "best_metrics" in ta and ta["best_metrics"]:
                    report.append(f"  Best F1 score: {ta['best_metrics']['f1']:.3f}")

        report.append("")
        report.append("=" * 80)

        # Save report
        with open(output_dir / "analysis_report_with_prototypes.txt", "w") as f:
            f.write("\n".join(report))

        # Print report
        print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Analysis of prototype ranking model outcomes."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., results/prototype_ranking/simple/run_20250729_145841_seed42)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number to use for model loading and inference (default: 1)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to use for visualization (default: 100)",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage training checkpoint paths (fold_X/checkpoint.pt). Default is two-stage (fold_X/finetuning/checkpoint.pt)",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return

    analyzer = PrototypeAnalyzer(run_dir)

    # Run analysis with specified parameters
    analyzer.run_analysis(
        fold=args.fold, sample_size=args.sample_size, is_single_stage=args.single_stage
    )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
