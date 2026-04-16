"""Package-owned embedding-alignment analysis helpers."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def entropy(input_array: np.ndarray) -> float:
    """Calculate Shannon entropy of an array."""
    if len(input_array) == 0:
        raise ValueError("Input array cannot be empty")

    _, counts = np.unique(input_array, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def generate_sphere_points(
    dim: int,
    num_points: int,
    radius: float = 1.0,
) -> np.ndarray:
    """Generate random points on a sphere surface."""
    if dim <= 0 or num_points <= 0:
        raise ValueError("Dimension and number of points must be positive")

    points = np.random.randn(num_points, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * radius


def lunif(x: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    """Calculate local uniformity using Gaussian potential."""
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 points to compute pairwise distances")
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()


def calculate_sample_alignment_distance(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate sample alignment distance (SAD) for multi-view embeddings."""
    if embeddings.ndim != 3:
        raise ValueError("SAD requires 3D embeddings with shape (N, num_variants, D)")
    if embeddings.shape[1] < 2:
        raise ValueError("SAD requires at least 2 variants per sample")

    n_samples, num_variants, _ = embeddings.shape
    if len(labels) != n_samples:
        raise ValueError("Number of embeddings must match number of labels")

    if num_variants == 2:
        variant_1 = embeddings[:, 0, :]
        variant_2 = embeddings[:, 1, :]
        all_distances = np.linalg.norm(variant_1 - variant_2, axis=1)
    else:
        all_distances = []
        for index in range(n_samples):
            from scipy.spatial.distance import pdist

            variant_distances = pdist(embeddings[index], metric="euclidean")
            all_distances.append(np.mean(variant_distances))
        all_distances = np.array(all_distances)

    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    sad_0 = all_distances[indices_0] if len(indices_0) > 0 else np.array([])
    sad_1 = all_distances[indices_1] if len(indices_1) > 0 else np.array([])
    return sad_0, sad_1


def calculate_sample_alignment_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Calculate sample alignment accuracy (SAA) for multi-view embeddings."""
    if embeddings.ndim != 3:
        raise ValueError("SAA requires 3D embeddings with shape (N, num_variants, D)")
    if embeddings.shape[1] < 2:
        raise ValueError("SAA requires at least 2 variants per sample")

    n_samples, _, _ = embeddings.shape
    if len(labels) != n_samples:
        raise ValueError("Number of embeddings must match number of labels")

    anchor_views = embeddings[:, 0, :]
    positive_distances = []
    for index in range(n_samples):
        anchor = anchor_views[index]
        other_variants = embeddings[index, 1:, :]
        if other_variants.shape[0] > 0:
            variant_distances = np.linalg.norm(other_variants - anchor, axis=1)
            positive_distances.append(np.mean(variant_distances))
        else:
            positive_distances.append(0.0)
    positive_distances = np.array(positive_distances)

    saa_scores = []
    for index in range(n_samples):
        anchor_view = embeddings[index, 0, :]
        negative_distances = []
        for other_index in range(n_samples):
            if index == other_index:
                continue
            negative_anchor = embeddings[other_index, 0, :]
            negative_distances.append(np.linalg.norm(anchor_view - negative_anchor))

        if negative_distances:
            min_negative_distance = min(negative_distances)
            positive_distance = positive_distances[index]
            saa_scores.append(1.0 if positive_distance < min_negative_distance else 0.0)
        else:
            saa_scores.append(0.0)

    saa_scores = np.array(saa_scores)
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    acc_cls0 = saa_scores[indices_0].mean() * 100.0 if len(indices_0) > 0 else 0.0
    acc_cls1 = saa_scores[indices_1].mean() * 100.0 if len(indices_1) > 0 else 0.0
    return acc_cls0, acc_cls1


def calculate_class_alignment_distance(
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
) -> Dict:
    """Calculate class alignment distance (CAD)."""
    try:
        if all_embeddings.ndim == 3:
            embeddings = all_embeddings.mean(axis=1)
        else:
            embeddings = all_embeddings

        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        elif hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        logger.debug(
            "CAD calculation: embeddings shape %s, labels shape %s",
            embeddings.shape,
            all_labels.shape,
        )

        results = {}
        for label in [0, 1]:
            mask = all_labels == label
            class_embeddings = embeddings[mask]
            n_samples = len(class_embeddings)
            logger.debug("CAD Class %s: %s samples", label, n_samples)

            if n_samples < 2:
                results[f"class_{label}"] = {"mean": 0.0, "std": 0.0}
                continue

            from scipy.spatial.distance import pdist

            distances = pdist(class_embeddings, metric="euclidean")
            if len(distances) > 0:
                cad_mean = np.mean(distances)
                cad_std = np.std(distances)
            else:
                cad_mean = 0.0
                cad_std = 0.0

            results[f"class_{label}"] = {
                "mean": float(cad_mean),
                "std": float(cad_std),
            }

        logger.debug("CAD results: %s", results)
        return results

    except Exception as exc:
        logger.error("Failed to calculate CAD: %s", exc)
        return {"error": f"Failed to calculate CAD: {exc}"}


def calculate_class_alignment_consistency(
    sim: np.ndarray,
    all_labels: np.ndarray,
) -> Dict:
    """Calculate class alignment consistency (CAC)."""
    try:
        results = {}
        for label in [0, 1]:
            indices = np.where(all_labels == label)[0]
            if len(indices) == 0:
                results[f"class_{label}"] = {"mean": 0.0, "std": 0.0}
                continue

            r = 0.01
            n_neighbor = max(1, int(sim.shape[0] * r))
            neighborhoods = np.argsort(-sim, axis=1)[:, :n_neighbor]

            cac_scores = np.zeros(len(indices))
            for score_index, sample_index in enumerate(indices):
                neighborhood = neighborhoods[sample_index]
                same_class_neighbors = np.sum(
                    all_labels[neighborhood] == all_labels[sample_index]
                )
                cac_scores[score_index] = same_class_neighbors / n_neighbor

            cac_percentages = cac_scores * 100.0
            results[f"class_{label}"] = {
                "mean": float(np.mean(cac_percentages)),
                "std": float(np.std(cac_percentages)),
            }

        logger.debug("CAC results: %s", results)
        return results

    except Exception as exc:
        logger.error("Failed to calculate CAC: %s", exc)
        return {"error": f"Failed to calculate CAC: {exc}"}


def calculate_gaussian_potential_uniformity(
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
) -> Dict:
    """Calculate Gaussian Potential Uniformity (GPU)."""
    try:
        if all_embeddings.ndim == 3:
            embeddings = all_embeddings.mean(axis=1)
        else:
            embeddings = all_embeddings

        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        elif hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        logger.debug(
            "GPU calculation: embeddings shape %s, labels shape %s",
            embeddings.shape,
            all_labels.shape,
        )
        logger.debug("Unique labels found: %s", np.unique(all_labels))

        results = {}
        for label in [0, 1]:
            mask = all_labels == label
            class_embeddings = embeddings[mask]
            n_samples = len(class_embeddings)
            logger.debug("Class %s: %s samples", label, n_samples)

            if n_samples < 2:
                results[f"class_{label}"] = 0.0
                continue

            from sklearn.metrics.pairwise import euclidean_distances

            distances_sq = euclidean_distances(class_embeddings, squared=True)
            gaussian_potentials = np.exp(-distances_sq)
            upper_tri_indices = np.triu_indices(n_samples, k=1)
            unique_pair_potentials = gaussian_potentials[upper_tri_indices]
            avg_potential = np.mean(unique_pair_potentials)
            gpu_score = np.log(avg_potential + 1e-8)
            results[f"class_{label}"] = float(gpu_score)

        overall_gpu = np.mean([results["class_0"], results["class_1"]])
        results["mean"] = float(overall_gpu)
        logger.debug("GPU results: %s", results)
        return results

    except Exception as exc:
        return {"error": f"Failed to calculate GPU: {exc}"}


def calculate_probabilistic_entropy_uniformity(
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    radius: float = 1.0,
    n_iterations: int = 5,
    multiplier: int = 10,
) -> Dict:
    """Calculate Probabilistic Entropy Uniformity (PEU)."""
    try:
        if all_embeddings.ndim == 3:
            embeddings = all_embeddings.mean(axis=1)
        else:
            embeddings = all_embeddings

        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        elif hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        logger.debug(
            "PEU calculation: embeddings shape %s, labels shape %s",
            embeddings.shape,
            all_labels.shape,
        )

        results = {}
        for label in [0, 1]:
            mask = all_labels == label
            class_embeddings = embeddings[mask]
            n_samples = len(class_embeddings)
            logger.debug("PEU Class %s: %s samples", label, n_samples)

            if n_samples < 2:
                results[f"class_{label}_mean"] = 0.0
                results[f"class_{label}_std"] = 0.0
                continue

            uniformity_entropies = []
            for iteration in range(n_iterations):
                try:
                    n_points = multiplier * n_samples
                    points = generate_sphere_points(
                        class_embeddings.shape[1],
                        n_points,
                        radius,
                    )

                    from sklearn.metrics.pairwise import euclidean_distances

                    distances = euclidean_distances(class_embeddings, points)
                    closest_point_indices = np.argmin(distances, axis=1)

                    if len(np.unique(closest_point_indices)) > 1:
                        actual_entropy = entropy(closest_point_indices)
                        max_entropy = entropy(np.arange(n_points))
                        peu_score = (
                            actual_entropy / max_entropy if max_entropy > 0 else 0.0
                        )
                    else:
                        peu_score = 0.0

                    uniformity_entropies.append(peu_score)
                except Exception as exc:
                    logger.warning(
                        "PEU iteration %s failed for class %s: %s",
                        iteration,
                        label,
                        exc,
                    )
                    uniformity_entropies.append(0.0)

            if uniformity_entropies:
                peu_mean = np.mean(uniformity_entropies) * 100
                peu_std = np.std(uniformity_entropies) * 100
            else:
                peu_mean = 0.0
                peu_std = 0.0

            results[f"class_{label}_mean"] = float(peu_mean)
            results[f"class_{label}_std"] = float(peu_std)

        results["overall_mean"] = float(
            np.mean([results["class_0_mean"], results["class_1_mean"]])
        )
        results["overall_std"] = float(
            np.mean([results["class_0_std"], results["class_1_std"]])
        )
        logger.debug("PEU results: %s", results)
        return results

    except Exception as exc:
        logger.error("Failed to calculate PEU: %s", exc)
        return {"error": f"Failed to calculate PEU: {exc}"}


def calculate_all_alignment_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_samples: Optional[int] = None,
) -> dict:
    """Calculate the full alignment metric bundle for embeddings and labels."""
    if embeddings.shape[0] != len(labels):
        raise ValueError("Number of embeddings must match number of labels")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty data")

    if embeddings.ndim == 2:
        averaged_embeddings = embeddings
        has_variants = False
        logger.debug("Processing baseline embeddings: %s", embeddings.shape)
    elif embeddings.ndim == 3:
        _, num_variants, _ = embeddings.shape
        if num_variants < 2:
            raise ValueError(
                "Expected at least 2 variants for contrastive learning, got "
                f"{num_variants}"
            )
        averaged_embeddings = embeddings.mean(axis=1)
        has_variants = True
        logger.debug(
            "Processing multi-view embeddings: %s, averaged to: %s",
            embeddings.shape,
            averaged_embeddings.shape,
        )
    else:
        raise ValueError(
            "Expected embeddings to be 2D [N, D] or 3D [N, num_variants, D], "
            f"got shape {embeddings.shape}"
        )

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(averaged_embeddings)
    results = {}

    try:
        results["gpu"] = calculate_gaussian_potential_uniformity(
            averaged_embeddings,
            labels,
        )
    except Exception as exc:
        results["gpu"] = {"error": str(exc)}

    try:
        results["peu"] = calculate_probabilistic_entropy_uniformity(
            averaged_embeddings,
            labels,
        )
    except Exception as exc:
        results["peu"] = {"error": str(exc)}

    try:
        results["cad"] = calculate_class_alignment_distance(
            averaged_embeddings,
            labels,
        )
    except Exception as exc:
        results["cad"] = {"error": str(exc)}

    try:
        if has_variants:
            n_samples, n_variants, dim = embeddings.shape
            all_views = embeddings.reshape(-1, dim)
            all_views_labels = np.repeat(labels, n_variants)
            all_views_similarities = cosine_similarity(all_views)
            cac_scores = calculate_class_alignment_consistency(
                all_views_similarities,
                all_views_labels,
            )
        else:
            cac_scores = calculate_class_alignment_consistency(similarities, labels)

        if "error" not in cac_scores:
            results["cac"] = cac_scores
        else:
            results["cac"] = {"error": "Failed to compute CAC"}
    except Exception as exc:
        results["cac"] = {"error": str(exc)}

    if has_variants:
        try:
            sad_0, sad_1 = calculate_sample_alignment_distance(embeddings, labels)
            if len(sad_0) > 0 and len(sad_1) > 0:
                results["sad"] = {
                    "class_0": {
                        "mean": float(np.mean(sad_0)),
                        "std": float(np.std(sad_0)),
                    },
                    "class_1": {
                        "mean": float(np.mean(sad_1)),
                        "std": float(np.std(sad_1)),
                    },
                }
            else:
                results["sad"] = {"error": "No samples in one or both classes"}
        except Exception as exc:
            results["sad"] = {"error": str(exc)}

        try:
            saa_0, saa_1 = calculate_sample_alignment_accuracy(embeddings, labels)
            results["saa"] = {
                "class_0": saa_0,
                "class_1": saa_1,
                "mean": (saa_0 + saa_1) / 2,
            }
        except Exception as exc:
            results["saa"] = {"error": str(exc)}
    else:
        results["variant_metrics"] = {
            "error": (
                "SAD/SAA require multi-view embeddings "
                "(not available for baseline models)"
            )
        }

    return results


__all__ = [
    "calculate_all_alignment_metrics",
    "calculate_class_alignment_consistency",
    "calculate_class_alignment_distance",
    "calculate_gaussian_potential_uniformity",
    "calculate_probabilistic_entropy_uniformity",
    "calculate_sample_alignment_accuracy",
    "calculate_sample_alignment_distance",
]
