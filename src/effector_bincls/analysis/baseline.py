# ruff: noqa: E501,I001,B905,F841
#!/usr/bin/env python3
"""
Analysis of baseline model outcomes.

This script provides insights into the retained SimplePredictor baseline trained
with BCE loss for binary classification, including:
1. Embedding space visualization using dimensionality reduction
2. Classification performance analysis
3. Feature space analysis
4. Model interpretability insights
5. Cross-fold consistency analysis
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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from effector_bincls.checkpoints import get_checkpoint_path
from effector_bincls.data import (
    DEFAULT_PARTITION_COLUMN,
    SimpleDataset,
    load_labeled_dataset,
    resolve_label_columns,
)
from effector_bincls.metrics import find_optimal_threshold
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


class BaselineAnalyzer:
    """
    Analyzer for baseline model outcomes with embedding visualization.
    """

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

    def generate_embeddings_with_model(
        self,
        fold: int = 1,
        sample_size: int = 100,
    ) -> Dict:
        """
        Generate embeddings by loading trained baseline models and running inference on OOF data.
        Then apply dimensionality reduction for visualization.

        Args:
            fold: Fold number to use for model loading and inference (default: 1)
            sample_size: Number of samples to use for visualization (default: 100)

        Returns:
            Dictionary containing:
            - embeddings_nd: Original embeddings (N x D)
            - embeddings_2d: 2D projections for visualization (N x 2)
            - labels: Labels for the sampled data (N,)
            - model_type: Type of model used
        """
        logger.info(
            f"Generating embeddings using trained baseline models (fold {fold}, sample_size {sample_size})..."
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
        test_csv_path = Path(config.data.csv_path)
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
                f"Unsupported model type '{model_type}' for baseline analysis."
            )

        dataset = SimpleDataset(
            embedding_dir=config.data.embedding_dir,
            csv_path=config.data.csv_path,
            sequence_ids=sampled_ids,
            normalize=getattr(config.features, "normalize", True),
            pooling_type=getattr(config.features, "pooling_type", "mean"),
            use_variants=False,
            label_config=label_config,
            logger=logger,
        )

        # Create data loader
        from torch.utils.data import DataLoader

        num_workers = int(getattr(config.hardware, "num_workers", 0))
        data_loader = DataLoader(
            dataset,
            batch_size=32,  # Smaller batch size for visualization
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        # Load the trained model for this fold using the existing infrastructure
        # Baseline models are always single-stage
        model_path = get_checkpoint_path(self.run_dir, fold, is_single_stage=True)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for fold {fold}: {model_path}")

        # Reuse the retained package-native checkpoint loader during the transition.
        from effector_bincls.evaluation.baseline import load_baseline_model

        logger.info(f"Loading baseline model from {model_path}")
        model = load_baseline_model(model_path, config, device)

        # Set model to finetuning mode (classification) if it supports it
        if hasattr(model, "set_training_mode"):
            model.set_training_mode("finetuning")
            logger.info(f"Set {type(model).__name__} to finetuning mode")

        logger.info(f"Successfully loaded baseline model checkpoint from {model_path}")
        logger.info(f"Model type: {type(model).__name__}")

        logger.info(
            f"Running inference for {sample_size} samples using fold {fold} model..."
        )

        # Run inference to get embeddings and logits
        all_embeddings = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                if isinstance(features, tuple):
                    features = tuple(f.to(device) for f in features)
                    # Use return_features=True to get intermediate representations
                    outputs = model(*features, return_features=True)
                else:
                    features = features.to(device)
                    # Use return_features=True to get intermediate representations
                    outputs = model(features, return_features=True)

                # Parse outputs based on model type and return_features flag
                if isinstance(outputs, tuple):
                    if len(outputs) == 2:
                        # (logits, features) format
                        logits, embeddings = outputs
                    elif len(outputs) == 3:
                        # (logits, contrastive_embeddings, features) format
                        logits, _, embeddings = outputs
                    else:
                        logger.warning(
                            f"Unexpected tuple output length: {len(outputs)}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Expected tuple output with features, got {type(outputs)}"
                    )
                    continue

                # Process embeddings consistently
                # Handle variant dimensions - use mean across variants for visualization
                if embeddings.dim() == 3:  # [batch_size, num_variants, feat_dim]
                    # Average across variants for visualization
                    embeddings = embeddings.mean(dim=1)  # [batch_size, feat_dim]
                    logger.debug(
                        f"Averaged {embeddings.shape[1]} variants for visualization"
                    )
                elif (
                    embeddings.dim() == 2
                ):  # [batch_size, feat_dim] - already single variant
                    pass  # No processing needed
                else:
                    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

                # Log embedding information for debugging
                logger.debug(f"Embedding shape: {embeddings.shape}")

                # Store embeddings, logits and labels
                all_embeddings.append(embeddings.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_embeddings:
            raise ValueError("No embeddings could be generated from the model")

        # Combine embeddings, logits and labels
        embeddings = np.concatenate(all_embeddings, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Flatten labels if they're 2D
        if labels.ndim > 1:
            labels = labels.flatten()

        logger.info(
            f"Generated {len(embeddings)} embeddings, shape: {embeddings.shape}"
        )
        logger.info(f"Labels shape: {labels.shape}")

        # Apply dimensionality reduction to embeddings
        logger.info("Applying UMAP dimensionality reduction...")

        # Check for any infinite or NaN values and clean data thoroughly
        if not np.isfinite(embeddings).all():
            logger.warning("Found non-finite values in data, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Additional data validation for UMAP compatibility
        if embeddings.shape[0] < 15:  # UMAP needs at least n_neighbors samples
            logger.warning("Too few samples for UMAP, using PCA instead")
            use_umap = False
        else:
            use_umap = True

        # Try UMAP, fallback to PCA if it fails
        if use_umap:
            try:
                # Fit UMAP on embeddings
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(
                        15, embeddings.shape[0] - 1
                    ),  # Ensure n_neighbors <= n_samples
                    min_dist=0.1,
                    random_state=42,
                    metric="cosine",
                )

                # Fit and transform
                embeddings_2d = reducer.fit_transform(embeddings)
                logger.info("Successfully applied UMAP dimensionality reduction")

            except Exception as e:
                logger.warning(f"UMAP failed: {e}, falling back to PCA")
                use_umap = False

        if not use_umap:
            # Fallback to PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            logger.info(
                f"Applied PCA dimensionality reduction, explained variance: {pca.explained_variance_ratio_.sum():.3f}"
            )

        # Build clean projection bundle
        projection = {
            "embeddings_nd": embeddings,
            "embeddings_2d": embeddings_2d,
            "logits": logits,
            "labels": labels,
            "model_type": type(model).__name__,
        }

        # Add variant embeddings if available for alignment metrics
        projection["variant_embeddings"] = None
        logger.info(
            "No variant embeddings available - SAD/SAA metrics will not be computed"
        )

        logger.info(
            f"Dimensionality reduction complete. Embeddings shape: {embeddings_2d.shape}"
        )

        return projection

    def visualize_embedding_space(
        self,
        projection: Dict,
        save_dir: Path,
    ):
        """
        Visualize the embedding space using UMAP projections.

        Args:
            projection: Dictionary containing embeddings in 2D and original space
            save_dir: Directory to save visualizations
        """
        logger.info("Creating embedding space visualizations...")

        embeddings_2d = projection["embeddings_2d"]
        labels = projection["labels"]
        logits = projection["logits"]
        model_type = projection["model_type"]

        # Create multiple visualization approaches
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Scatter plot with class colors
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

        axes[0, 0].set_title(f"UMAP Embedding Space - {model_type}")
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

        axes[0, 1].set_title("Density Plot")
        axes[0, 1].set_xlabel("UMAP Component 1")
        axes[0, 1].set_ylabel("UMAP Component 2")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Logit-based visualization
        # Color points based on logit values
        scatter = axes[0, 2].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=logits.flatten(),
            cmap="RdBu_r",
            alpha=0.7,
            s=30,
        )

        axes[0, 2].set_title("Logit Values")
        axes[0, 2].set_xlabel("UMAP Component 1")
        axes[0, 2].set_ylabel("UMAP Component 2")
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label="Logit Value")

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

            axes[1, 0].set_title("Class Centroids")
            axes[1, 0].set_xlabel("UMAP Component 1")
            axes[1, 0].set_ylabel("UMAP Component 2")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Feature space analysis
        # Note: Embeddings are L2 normalized for analysis, so all have unit norm
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Embeddings are L2 normalized\nfor alignment metrics analysis",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        # 6. Summary statistics
        axes[1, 2].axis("off")
        stats_text = f"""
        Baseline Model Analysis
        
        Model type: {model_type}
        Total samples: {len(embeddings_2d)}
        Positive samples: {np.sum(pos_mask)}
        Negative samples: {np.sum(neg_mask)}
        
        Feature Space Analysis:
        - Embedding dimension: {projection["embeddings_nd"].shape[1]}
        - Embeddings are L2 normalized for analysis
        
        Logit Analysis:
        - Mean positive logits: {np.mean(logits[pos_mask]):.3f}
        - Mean negative logits: {np.mean(logits[neg_mask]):.3f}
        - Logit separation: {np.mean(logits[pos_mask]) - np.mean(logits[neg_mask]):.3f}
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

    def analyze_feature_space(self, projection: Dict) -> Dict:
        """
        Analyze the learned feature space characteristics.

        Args:
            projection: Dictionary containing embeddings and labels

        Returns:
            Dictionary with feature space analysis results
        """
        logger.info("Analyzing feature space characteristics...")

        embeddings = projection["embeddings_nd"]
        labels = projection["labels"]
        logits = projection["logits"]

        # Separate by class
        pos_mask = labels == 1
        neg_mask = labels == 0

        # Calculate cosine similarities between samples
        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]

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

        # Logit analysis
        pos_logits = logits[pos_mask].flatten()
        neg_logits = logits[neg_mask].flatten()

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
            "logit_analysis": {
                "pos_mean": float(np.mean(pos_logits)),
                "pos_std": float(np.std(pos_logits)),
                "neg_mean": float(np.mean(neg_logits)),
                "neg_std": float(np.std(neg_logits)),
                "separation": float(np.mean(pos_logits) - np.mean(neg_logits)),
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
                f"Positive class coherence: {analysis['cosine_similarities']['pos_intra_mean']:.3f}"
            )
        if len(neg_intra_sim) > 0:
            logger.info(
                f"Negative class coherence: {analysis['cosine_similarities']['neg_intra_mean']:.3f}"
            )
        if len(inter_sim) > 0:
            logger.info(
                f"Inter-class separation: {analysis['class_separation']['inter_class_distance']:.3f}"
            )

        logger.info(f"Logit separation: {analysis['logit_analysis']['separation']:.3f}")

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
            projection: Dictionary containing embeddings and labels from generate_embeddings_with_model
            n_samples: Number of samples per batch (for SAD/SAA metrics)

        Returns:
            Dictionary containing all alignment metrics
        """
        logger.info("Analyzing embedding space alignment metrics...")

        embeddings = projection["embeddings_nd"]
        labels = projection["labels"]

        if embeddings.shape[0] != len(labels):
            logger.error("Embeddings and labels shape mismatch")
            return {}

        logger.info(
            f"Computing alignment metrics for {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions"
        )

        try:
            # Normalize embeddings before computing alignment metrics
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / (norms + 1e-8)

            logger.info("Applied L2 normalization to embeddings for alignment metrics")

            # Calculate all alignment metrics using normalized embeddings
            alignment_metrics = calculate_all_alignment_metrics(
                embeddings=normalized_embeddings, labels=labels
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

    def analyze_threshold_performance(self) -> Dict:
        """
        Analyze threshold performance using out-of-fold predictions.

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

        # Reconstruct predictions and labels from the analysis
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
                threshold_analysis["best_metrics"]["optimal_threshold"]
                if threshold_analysis["best_metrics"]
                else None
            ),
            threshold_method_used=threshold_analysis["best_method"],
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

    def plot_feature_space_analysis(self, feature_analysis: Dict, save_dir: Path):
        """Plot feature space analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Cosine similarities
        sim_stats = feature_analysis["cosine_similarities"]
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

        # 2. Logit analysis
        logit_stats = feature_analysis["logit_analysis"]
        logit_categories = ["Positive", "Negative", "Separation"]
        logit_values = [
            logit_stats["pos_mean"],
            logit_stats["neg_mean"],
            logit_stats["separation"],
        ]
        logit_colors = ["green", "red", "blue"]

        bars = axes[0, 1].bar(
            logit_categories, logit_values, color=logit_colors, alpha=0.7
        )
        axes[0, 1].set_title("Logit Analysis")
        axes[0, 1].set_ylabel("Logit Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, logit_values):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 3. Class separation metrics
        sep_stats = feature_analysis["class_separation"]
        sep_categories = ["Pos Coherence", "Neg Coherence", "Inter Distance"]
        sep_values = [
            sep_stats["pos_cluster_coherence"],
            sep_stats["neg_cluster_coherence"],
            sep_stats["inter_class_distance"],
        ]
        sep_colors = ["green", "red", "blue"]

        bars = axes[1, 0].bar(sep_categories, sep_values, color=sep_colors, alpha=0.7)
        axes[1, 0].set_title("Class Separation Metrics")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, sep_values):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            save_dir / "feature_space_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Saved feature space analysis to {save_dir / 'feature_space_analysis.png'}"
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

    def run_analysis(
        self,
        output_dir: Path = None,
        fold: int = 1,
        sample_size: int = 100,
    ):
        """Run analysis for baseline models.

        Args:
            output_dir: Directory to save analysis results
            fold: Fold number to use for model loading and inference
            sample_size: Number of samples to use for visualization
        """
        if output_dir is None:
            output_dir = self.run_dir / "baseline_analysis"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting baseline analysis (fold {fold}, sample_size {sample_size}), saving to {output_dir}"
        )

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
        logger.info(f"Negative samples: {np.sum(labels == 0)}")

        # 1. Generate embeddings and 2D projections
        projection = self.generate_embeddings_with_model(
            fold=fold,
            sample_size=sample_size,
        )

        # 2. Visualize embedding space
        self.visualize_embedding_space(projection, output_dir)

        # 3. Analyze feature space characteristics
        feature_analysis = self.analyze_feature_space(projection)
        self.plot_feature_space_analysis(feature_analysis, output_dir)

        # 4. NEW: Analyze embedding space alignment metrics
        alignment_analysis = self.analyze_embedding_space_alignment(projection)
        self.plot_alignment_metrics(alignment_analysis, output_dir)

        # 5. Threshold analysis using out-of-fold predictions
        threshold_analysis = self.analyze_threshold_performance()
        self.plot_threshold_analysis(threshold_analysis, output_dir)

        # 5. Save comprehensive results
        summary = {
            "embedding_analysis": {
                "n_samples": len(predictions),
                "n_positive": np.sum(labels == 1),
                "n_negative": np.sum(labels == 0),
                "model_type": projection["model_type"],
            },
            "projection": {
                "embeddings_shape": projection["embeddings_nd"].shape,
                "embeddings_2d_shape": projection["embeddings_2d"].shape,
            },
            "feature_analysis": feature_analysis,
            "alignment_analysis": alignment_analysis,  # NEW: Add alignment analysis
            "threshold_analysis": threshold_analysis,
            "analysis_parameters": {
                "fold": fold,
                "sample_size": sample_size,
            },
        }

        import json

        with open(output_dir / "baseline_analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 6. Create summary report
        self.create_summary_report(summary, output_dir)

        logger.info(f"Baseline analysis complete! Results saved to {output_dir}")
        return summary

    def create_summary_report(self, summary: Dict, output_dir: Path):
        """Create a text summary report for baseline analysis."""
        report = []
        report.append("=" * 80)
        report.append("BASELINE MODEL ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Model information
        report.append("MODEL INFORMATION:")
        report.append("-" * 40)
        report.append(f"Model type: {summary['embedding_analysis']['model_type']}")
        report.append(f"Total samples: {summary['embedding_analysis']['n_samples']}")
        report.append(
            f"Positive samples: {summary['embedding_analysis']['n_positive']}"
        )
        report.append(
            f"Negative samples: {summary['embedding_analysis']['n_negative']}"
        )
        report.append("")

        # Feature space analysis
        report.append("FEATURE SPACE ANALYSIS:")
        report.append("-" * 40)
        fa = summary["feature_analysis"]

        # Cosine similarities
        sims = fa["cosine_similarities"]
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

        # Logit analysis
        logits = fa["logit_analysis"]
        report.append("Logit Analysis:")
        report.append(
            f"  Positive mean: {logits['pos_mean']:.3f} ± {logits['pos_std']:.3f}"
        )
        report.append(
            f"  Negative mean: {logits['neg_mean']:.3f} ± {logits['neg_std']:.3f}"
        )
        report.append(f"  Separation: {logits['separation']:.3f}")
        report.append("")

        # Class separation
        sep = fa["class_separation"]
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

        # Threshold analysis
        if "threshold_analysis" in summary:
            report.append("THRESHOLD ANALYSIS:")
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

        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)

        # Feature space insights
        if sep["pos_cluster_coherence"] > 0.7:
            report.append("✓ High positive class coherence")
        else:
            report.append("⚠ Room for improvement in positive class coherence")

        if sep["neg_cluster_coherence"] > 0.7:
            report.append("✓ High negative class coherence")
        else:
            report.append("⚠ Room for improvement in negative class coherence")

        if sep["inter_class_distance"] > 0.3:
            report.append("✓ Good inter-class separation")
        else:
            report.append("⚠ Room for improvement in inter-class separation")

        # Logit insights
        if logits["separation"] > 2.0:
            report.append("✓ Strong logit separation between classes")
        elif logits["separation"] > 1.0:
            report.append("✓ Moderate logit separation between classes")
        else:
            report.append("⚠ Weak logit separation between classes")

        # Threshold analysis insights
        if "threshold_analysis" in summary:
            ta = summary["threshold_analysis"]
            if "best_method" in ta and ta["best_method"]:
                report.append(f"✓ Best threshold method: {ta['best_method']}")
                if "best_metrics" in ta and ta["best_metrics"]:
                    report.append(f"  Best F1 score: {ta['best_metrics']['f1']:.3f}")

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

        report.append("")
        report.append("=" * 80)

        # Save report
        with open(output_dir / "baseline_analysis_report.txt", "w") as f:
            f.write("\n".join(report))

        # Print report
        print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Analysis of baseline model outcomes.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., results/baseline_bce/run_20250729_145841_seed42)",
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

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return

    analyzer = BaselineAnalyzer(run_dir)

    # Run analysis with specified parameters
    analyzer.run_analysis(fold=args.fold, sample_size=args.sample_size)

    logger.info("Baseline analysis complete!")


if __name__ == "__main__":
    main()
