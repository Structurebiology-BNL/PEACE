"""Package-owned contrastive and prototype loss implementations."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from effector_bincls.prototype_scoring import compute_prototype_distance_scores


class ConSupPrototypeLoss(nn.Module):
    """
    Improved ConSup loss with prototype alignment and unsupervised contrastive learning.

    Key improvements:
    1. Consistent normalization across all similarity computations
    2. Simplified margin conditions with single eps parameter
    3. Enhanced numerical stability
    4. Better gradient flow management
    """

    def __init__(
        self,
        temperature: float = 0.07,
        eps: float = 0.1,
        prototype_weight: float = 1.0,
        unsupervised_weight: float = 1.0,
        device: Optional[torch.device] = None,
        logger=None,
        eps_pos: Optional[float] = None,
        eps_neg: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.prototype_weight = prototype_weight
        self.unsupervised_weight = unsupervised_weight
        self.device = device

        if logger is not None:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.eps_pos = eps_pos if eps_pos is not None else eps
        self.eps_neg = eps_neg if eps_neg is not None else eps
        self.logger.info(
            "Initialized ConSupPrototypeLoss with eps_pos=%.4f, eps_neg=%.4f",
            self.eps_pos,
            self.eps_neg,
        )

        self.prototypes = None
        self.eps_numerical = 1e-6

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the class prototypes to use in the loss calculation."""
        try:
            self.prototypes = prototypes.to(self.device)
            if torch.isnan(self.prototypes).any() or torch.isinf(self.prototypes).any():
                raise ValueError("Prototypes contain NaN or Inf values")
        except Exception as exc:
            self.logger.error("Failed to set prototypes: %s", exc)
            raise

    def _compute_unsupervised_contrastive_loss(
        self,
        features,
        batch_size,
        n_views,
    ):
        """Compute unsupervised contrastive loss with improved numerical stability."""
        if n_views < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        try:
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

            if (
                torch.isnan(contrast_feature).any()
                or torch.isinf(contrast_feature).any()
            ):
                self.logger.warning(
                    "Features contain NaN or Inf values, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            sim_matrix = (
                torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
            )
            sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            logits = sim_matrix - sim_matrix_max.detach()

            labels = torch.arange(batch_size).repeat(n_views).to(self.device)
            mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
            self_mask = torch.eye(
                batch_size * n_views, device=self.device, dtype=torch.bool
            )
            mask = mask * (~self_mask).float()

            num_positives_per_sample = mask.sum(1)
            if (num_positives_per_sample == 0).all():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            exp_logits = torch.exp(logits)
            exp_logits = exp_logits * (~self_mask).float()
            log_prob = logits - torch.log(
                exp_logits.sum(1, keepdim=True) + self.eps_numerical
            )

            valid_samples = num_positives_per_sample > 0
            if not valid_samples.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            mean_log_prob_pos = (mask * log_prob).sum(
                1
            ) / num_positives_per_sample.clamp(min=self.eps_numerical)
            valid_mean_log_prob_pos = mean_log_prob_pos[valid_samples]
            loss = -valid_mean_log_prob_pos.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    "Contrastive loss is NaN or Inf, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return loss

        except Exception as exc:
            self.logger.error("Error in contrastive loss computation: %s", exc)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _compute_prototype_alignment_loss(
        self,
        features,
        labels,
        batch_size,
        n_views,
    ):
        """Compute prototype alignment loss with improved stability."""
        if self.prototypes is None:
            raise ValueError("Prototypes must be set before computing alignment loss.")

        try:
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

            if (
                torch.isnan(contrast_feature).any()
                or torch.isinf(contrast_feature).any()
            ):
                self.logger.warning(
                    "Features contain NaN or Inf values, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            sim_to_prototypes = torch.matmul(contrast_feature, self.prototypes.T)
            sim_to_p0 = sim_to_prototypes[:, 0]
            sim_to_p1 = sim_to_prototypes[:, 1]

            if labels.dim() == 1:
                class_indices = labels
                if not torch.all((class_indices == 0) | (class_indices == 1)):
                    raise ValueError(
                        "Labels must contain only 0 or 1, "
                        f"got: {torch.unique(class_indices)}"
                    )
            elif labels.dim() == 2:
                if labels.shape[1] != 2:
                    raise ValueError(
                        "Expected 2 classes in one-hot labels, "
                        f"got shape: {labels.shape}"
                    )
                class_indices = labels.argmax(dim=1)
            else:
                raise ValueError(
                    "Unexpected labels dimension: "
                    f"{labels.dim()}, shape: {labels.shape}"
                )

            if class_indices.dim() > 1:
                class_indices = class_indices.squeeze()

            repeated_labels = (
                class_indices if n_views == 1 else class_indices.repeat(n_views)
            )
            is_class_0 = repeated_labels == 0
            is_class_1 = repeated_labels == 1
            pull_mask_0 = is_class_0 & (sim_to_p0 <= sim_to_p1 + self.eps_neg)
            pull_mask_1 = is_class_1 & (sim_to_p1 <= sim_to_p0 + self.eps_pos)
            pull_mask = (pull_mask_0 | pull_mask_1).float()

            proto_logits = sim_to_prototypes / self.temperature
            log_softmax_logits = F.log_softmax(proto_logits, dim=1)
            nll_loss = -log_softmax_logits[
                torch.arange(batch_size * n_views), repeated_labels
            ]

            num_pulled = pull_mask.sum().clamp(min=self.eps_numerical)
            if num_pulled == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            loss = (nll_loss * pull_mask).sum() / num_pulled
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    "Prototype alignment loss is NaN or Inf, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss

        except Exception as exc:
            self.logger.error("Error in prototype alignment loss computation: %s", exc)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute the improved ConSup loss."""
        if labels is None:
            raise ValueError("Labels must be provided for ConSup loss.")

        if self.device is None:
            self.device = features.device

        try:
            batch_size, n_views, _ = features.shape
            loss_components = {}
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if n_views > 1 and self.unsupervised_weight > 0:
                contrastive_loss = self._compute_unsupervised_contrastive_loss(
                    features,
                    batch_size,
                    n_views,
                )
                total_loss = total_loss + self.unsupervised_weight * contrastive_loss
                loss_components["contrastive"] = contrastive_loss.detach()

            if self.prototype_weight > 0:
                prototype_loss = self._compute_prototype_alignment_loss(
                    features,
                    labels,
                    batch_size,
                    n_views,
                )
                total_loss = total_loss + self.prototype_weight * prototype_loss
                loss_components["prototype"] = prototype_loss.detach()

            loss_components["total"] = total_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error(
                    "Total loss is NaN or Inf, returning zero loss components"
                )
                return {
                    "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "contrastive": torch.tensor(0.0, device=self.device),
                    "prototype": torch.tensor(0.0, device=self.device),
                }

            return loss_components

        except Exception as exc:
            self.logger.error("Error in forward pass: %s", exc)
            return {
                "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                "contrastive": torch.tensor(0.0, device=self.device),
                "prototype": torch.tensor(0.0, device=self.device),
            }


class PrototypeBCELoss(ConSupPrototypeLoss):
    """Prototype-based BCE loss with prototype alignment and BCE classification."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        unsupervised_weight: float = 0.5,
        prototype_weight: float = 1.0,
        scoring_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            unsupervised_weight=unsupervised_weight,
            prototype_weight=prototype_weight,
            **kwargs,
        )

        self.bce_weight = bce_weight
        self.scoring_temperature = scoring_temperature

        if bce_weight > 0:
            self.logger.info(
                "Extended ConSupPrototypeLoss for classification: "
                "bce_weight=%s, unsupervised_weight=%s, scoring_temperature=%s",
                bce_weight,
                unsupervised_weight,
                scoring_temperature,
            )
        else:
            self.logger.info(
                "Extended ConSupPrototypeLoss for classification: DISABLED "
                "(weight=0), unsupervised_weight=%s, scoring_temperature=%s",
                unsupervised_weight,
                scoring_temperature,
            )

    def _compute_classification_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Compute classification scores from prototype distances using all variants."""
        if self.prototypes is None:
            raise ValueError("Prototypes must be set")

        return compute_prototype_distance_scores(
            embeddings=features,
            prototypes=self.prototypes,
            scoring_temperature=self.scoring_temperature,
            logger=self.logger,
        )

    def _compute_bce_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy with logits on prototype-difference scores."""
        try:
            if scores.dim() > 1:
                scores = scores.squeeze()
            if labels.dim() > 1:
                labels = labels.squeeze()

            targets = labels.float()
            loss = F.binary_cross_entropy_with_logits(scores, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning("BCE loss is NaN/Inf; returning zero")
                return torch.tensor(0.0, device=scores.device, requires_grad=True)
            return loss
        except Exception as exc:
            self.logger.error("Error in BCE loss computation: %s", exc)
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute prototype BCE loss."""
        if labels is None:
            raise ValueError("Labels required for classification loss")

        if self.device is None:
            self.device = features.device

        try:
            batch_size, n_views, _ = features.shape
            loss_components = {}
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                self.logger.debug(
                    "PrototypeBCELoss: batch_size=%s, n_views=%s, "
                    "unsupervised_weight=%s",
                    batch_size,
                    n_views,
                    self.unsupervised_weight,
                )

            if self.prototype_weight > 0:
                prototype_loss = self._compute_prototype_alignment_loss(
                    features,
                    labels,
                    batch_size,
                    n_views,
                )
                total_loss = total_loss + self.prototype_weight * prototype_loss
                loss_components["prototype"] = prototype_loss.detach()

            if self.unsupervised_weight > 0 and n_views > 1:
                contrastive_loss = self._compute_unsupervised_contrastive_loss(
                    features,
                    batch_size,
                    n_views,
                )
                total_loss = total_loss + self.unsupervised_weight * contrastive_loss
                loss_components["contrastive"] = contrastive_loss.detach()

            if self.bce_weight > 0:
                if labels.dim() == 2:
                    class_indices = labels.argmax(dim=1)
                elif labels.dim() > 1:
                    class_indices = labels.squeeze()
                else:
                    class_indices = labels

                classification_scores = self._compute_classification_scores(features)
                if not classification_scores.requires_grad:
                    if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                        self.logger.debug(
                            "Classification scores do not require gradients "
                            "(possibly during validation)"
                        )

                bce_loss = self._compute_bce_loss(classification_scores, class_indices)
                if not bce_loss.requires_grad:
                    if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                        self.logger.debug(
                            "BCE loss does not require gradients "
                            "(possibly during validation)"
                        )

                weighted_bce_loss = self.bce_weight * bce_loss
                loss_components["bce"] = bce_loss.detach()
                total_loss = total_loss + weighted_bce_loss
            else:
                loss_components["bce"] = torch.tensor(0.0, device=self.device)

            loss_components["total"] = total_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error(
                    "Total prototype BCE loss is NaN or Inf, "
                    "returning zero loss components"
                )
                return {
                    "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "prototype": torch.tensor(0.0, device=self.device),
                    "contrastive": torch.tensor(0.0, device=self.device),
                    "bce": torch.tensor(0.0, device=self.device),
                }

            return loss_components

        except Exception as exc:
            self.logger.error("Error in prototype BCE loss forward pass: %s", exc)
            return {
                "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                "prototype": torch.tensor(0.0, device=self.device),
                "contrastive": torch.tensor(0.0, device=self.device),
                "bce": torch.tensor(0.0, device=self.device),
            }


class HybridContrastiveLoss(PrototypeBCELoss):
    """Hybrid contrastive learning loss with class-specific strategies."""

    def __init__(
        self,
        temperature: float = 0.07,
        eps: float = 0.1,
        minority_cls: int = 1,
        supervised_weight: float = 1.0,
        unsupervised_weight: float = 1.0,
        prototype_weight: float = 1.0,
        bce_weight: float = 1.0,
        device: Optional[torch.device] = None,
        logger=None,
        **kwargs,
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            prototype_weight=prototype_weight,
            unsupervised_weight=0.0,
            bce_weight=bce_weight,
            device=device,
            logger=logger,
            **kwargs,
        )

        self.minority_cls = minority_cls
        self.majority_cls = 1 - minority_cls
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight

        if self.bce_weight > 0:
            self.logger.info(
                "Extended PrototypeBCELoss for hybrid learning: bce_weight=%s",
                bce_weight,
            )
        else:
            self.logger.info(
                "HybridContrastiveLoss: BCE classification DISABLED (weight=0), "
                "prototype_alignment=ALWAYS_ENABLED"
            )

    def _compute_supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        minority_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss for minority class samples."""
        batch_size, n_views, feat_dim = features.shape

        if minority_mask.sum() < 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        sim_matrix = (
            torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        )

        if labels.dim() == 1:
            class_indices = labels
            if not torch.all((class_indices == 0) | (class_indices == 1)):
                raise ValueError(
                    "Labels must contain only 0 or 1, "
                    f"got: {torch.unique(class_indices)}"
                )
        elif labels.dim() == 2:
            if labels.shape[1] != 2:
                raise ValueError(
                    f"Expected 2 classes in one-hot labels, got shape: {labels.shape}"
                )
            class_indices = labels.argmax(dim=1)
        else:
            raise ValueError(
                f"Unexpected labels dimension: {labels.dim()}, shape: {labels.shape}"
            )

        repeated_class_indices = class_indices.repeat(n_views)
        repeated_minority_mask = minority_mask.repeat(n_views)

        self_mask = torch.eye(
            batch_size * n_views, device=self.device, dtype=torch.bool
        )

        positive_mask = torch.eq(
            repeated_class_indices.view(-1, 1), repeated_class_indices.view(1, -1)
        ).float()
        positive_mask = positive_mask * (~self_mask).float()

        valid_samples = torch.sum(positive_mask, dim=1) > 0
        if not torch.any(valid_samples):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        log_probs = torch.log_softmax(sim_matrix, dim=1)
        log_probs_pos = log_probs * positive_mask
        num_positives = positive_mask.sum(dim=1, keepdim=True).float()
        num_positives = torch.max(num_positives, torch.ones_like(num_positives))
        per_sample_loss = -log_probs_pos.sum(dim=1) / num_positives.squeeze()

        minority_valid_mask = valid_samples & repeated_minority_mask
        per_sample_loss = per_sample_loss * minority_valid_mask.float()
        if minority_valid_mask.sum() > 0:
            return per_sample_loss.sum() / minority_valid_mask.sum()
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute the hybrid contrastive loss with optional BCE classification."""
        if labels is None:
            raise ValueError("Labels required for hybrid loss")

        if self.device is None:
            self.device = features.device

        batch_size, n_views, _ = features.shape
        loss_components = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if labels.dim() == 2:
            class_indices = labels.argmax(dim=1)
        elif labels.dim() > 1:
            class_indices = labels.squeeze()
        else:
            class_indices = labels

        minority_mask = class_indices == self.minority_cls
        majority_mask = class_indices == self.majority_cls

        if self.prototype_weight > 0:
            prototype_loss = self._compute_prototype_alignment_loss(
                features,
                labels,
                batch_size,
                n_views,
            )
            total_loss = total_loss + self.prototype_weight * prototype_loss
            loss_components["prototype"] = prototype_loss.detach()

        if self.unsupervised_weight > 0 and torch.any(majority_mask) and n_views > 1:
            majority_features = features[majority_mask]
            if majority_features.size(0) >= 2:
                n_majority, n_views, _ = majority_features.shape
                unsupervised_loss = self._compute_unsupervised_contrastive_loss(
                    majority_features,
                    n_majority,
                    n_views,
                )
                total_loss = total_loss + self.unsupervised_weight * unsupervised_loss
                loss_components["unsupervised"] = unsupervised_loss.detach()

        if self.supervised_weight > 0 and torch.any(minority_mask) and n_views > 1:
            supervised_loss = self._compute_supervised_contrastive_loss(
                features,
                labels,
                minority_mask,
            )
            total_loss = total_loss + self.supervised_weight * supervised_loss
            loss_components["supervised"] = supervised_loss.detach()

        if self.bce_weight > 0:
            classification_scores = self._compute_classification_scores(features)
            bce_loss = self._compute_bce_loss(classification_scores, class_indices)
            total_loss = total_loss + self.bce_weight * bce_loss
            loss_components["bce"] = bce_loss.detach()
        else:
            loss_components["bce"] = torch.tensor(0.0, device=self.device)

        loss_components["total"] = total_loss
        return loss_components


__all__ = [
    "ConSupPrototypeLoss",
    "HybridContrastiveLoss",
    "PrototypeBCELoss",
]
