"""Package-owned prototype scoring helpers."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F


def compute_prototype_distance_scores(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    scoring_temperature: float = 1.0,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Compute prototype distance scores for 2D or 3D embeddings."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if prototypes is None:
        raise ValueError("Prototypes must be provided")
    if prototypes.shape[0] != 2:
        raise ValueError(
            f"Expected 2 prototypes (negative, positive), got {prototypes.shape[0]}"
        )

    try:
        if embeddings.dim() == 3:
            batch_size = embeddings.shape[0]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=2)
            if (
                torch.isnan(normalized_embeddings).any()
                or torch.isinf(normalized_embeddings).any()
            ):
                logger.warning("Normalized embeddings contain NaN or Inf values")
                return torch.zeros(
                    batch_size,
                    device=embeddings.device,
                    requires_grad=embeddings.requires_grad,
                )

            similarities = torch.matmul(normalized_embeddings, prototypes.T)
            scaled_similarities = similarities / scoring_temperature
            distance_scores = (
                scaled_similarities[:, :, 1] - scaled_similarities[:, :, 0]
            )
            return distance_scores.mean(dim=1)

        if embeddings.dim() == 2:
            batch_size = embeddings.shape[0]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            if (
                torch.isnan(normalized_embeddings).any()
                or torch.isinf(normalized_embeddings).any()
            ):
                logger.warning("Normalized embeddings contain NaN or Inf values")
                return torch.zeros(
                    batch_size,
                    device=embeddings.device,
                    requires_grad=embeddings.requires_grad,
                )

            similarities = torch.matmul(normalized_embeddings, prototypes.T)
            scaled_similarities = similarities / scoring_temperature
            return scaled_similarities[:, 1] - scaled_similarities[:, 0]

        raise ValueError(
            f"Expected embeddings to be 2D or 3D, got shape: {embeddings.shape}"
        )
    except Exception as exc:
        logger.error("Error in prototype ranking scores computation: %s", exc)
        return torch.zeros(
            embeddings.shape[0],
            device=embeddings.device,
            requires_grad=embeddings.requires_grad,
        )


def compute_prototype_probabilities(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    scoring_temperature: float = 1.0,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Convert prototype distance scores into sigmoid probabilities."""
    distance_scores = compute_prototype_distance_scores(
        embeddings=embeddings,
        prototypes=prototypes,
        scoring_temperature=scoring_temperature,
        logger=logger,
    )
    return torch.sigmoid(distance_scores)


__all__ = [
    "compute_prototype_distance_scores",
    "compute_prototype_probabilities",
]
