"""Trainer for BCE classification with multi-view contrastive regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from effector_bincls.training.trainers import BaselineTrainer


class ContrastiveBCETrainer(BaselineTrainer):
    """Train classification logits jointly with dropout-view InfoNCE."""

    def __init__(
        self,
        model,
        config,
        device="cuda",
        save_checkpoints=False,
        logger=None,
    ):
        super().__init__(
            model=model,
            config=config,
            device=device,
            save_checkpoints=save_checkpoints,
            logger=logger,
        )
        if not getattr(model, "use_contrastive", False):
            raise ValueError("ContrastiveBCETrainer requires use_contrastive=True")
        self.bce_weight = float(config.bce_weight)
        self.unsupervised_weight = float(config.unsupervised_weight)
        self.temperature = float(config.temperature)
        self.eps_numerical = 1e-6

    def _extract_logits(self, outputs) -> torch.Tensor:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError(
                "ContrastiveBCETrainer expects model output "
                "(logits, contrastive_embeddings)."
            )
        logits = outputs[0]
        if not torch.is_tensor(logits):
            raise ValueError(f"Expected tensor logits from model, got {type(logits)}")
        return logits

    def _compute_unsupervised_contrastive_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError(
                "Contrastive-BCE requires 3D contrastive embeddings shaped "
                "[batch, views, dim]."
            )
        batch_size, num_views, _ = embeddings.shape
        if num_views < 2:
            raise ValueError(
                "Contrastive-BCE requires at least two views per sequence."
            )

        features = torch.cat(torch.unbind(embeddings, dim=1), dim=0)
        features = F.normalize(features, p=2, dim=1)
        similarities = torch.matmul(features, features.T) / self.temperature
        logits = (
            similarities
            - similarities.max(
                dim=1,
                keepdim=True,
            ).values.detach()
        )

        sample_ids = torch.arange(batch_size, device=self.device).repeat(num_views)
        self_mask = torch.eye(
            batch_size * num_views,
            device=self.device,
            dtype=torch.bool,
        )
        positive_mask = sample_ids[:, None].eq(sample_ids[None, :]) & ~self_mask
        exp_logits = torch.exp(logits) * (~self_mask)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + self.eps_numerical
        )
        positive_count = positive_mask.sum(dim=1)
        mean_log_probability = (positive_mask * log_prob).sum(
            dim=1
        ) / positive_count.clamp(min=1)
        return -mean_log_probability.mean()

    def compute_loss(self, outputs, labels, is_training=True):
        logits = self._extract_logits(outputs).reshape(-1)
        labels = labels.reshape(-1).float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        contrastive_loss = self._compute_unsupervised_contrastive_loss(outputs[1])
        total_loss = (
            self.bce_weight * bce_loss + self.unsupervised_weight * contrastive_loss
        )
        return {
            "total": total_loss,
            "bce": bce_loss.detach(),
            "contrastive": contrastive_loss.detach(),
        }


__all__ = ["ContrastiveBCETrainer"]
