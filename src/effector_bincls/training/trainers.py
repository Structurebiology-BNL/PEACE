"""Package-owned training implementations for the retained workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from effector_bincls.metrics import find_optimal_threshold, multi_scores
from effector_bincls.prototype_scoring import compute_prototype_distance_scores
from effector_bincls.training.losses import (
    ConSupPrototypeLoss,
    HybridContrastiveLoss,
    PrototypeBCELoss,
)
from effector_bincls.training.runtime import BaseTrainer


def _config_get(section, key, default=None):
    """Read config values across dict-like and attribute-style config objects."""
    if section is None:
        return default
    if hasattr(section, key):
        return getattr(section, key)
    if isinstance(section, dict):
        return section.get(key, default)
    if hasattr(section, "get"):
        return section.get(key, default)
    return default


class BaselineTrainer(BaseTrainer):
    """Trainer for the retained BCE baseline workflow."""

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

        if not hasattr(model, "classification_head"):
            raise ValueError("Model must be SimplePredictor with classification_head")

        if hasattr(model, "set_training_mode"):
            model.set_training_mode("finetuning")
            self.logger.info("Set %s to finetuning mode", model.__class__.__name__)

        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        self.logger.info(
            "%s model: %s / %s parameters trainable (%.1f%%)",
            model.__class__.__name__,
            f"{trainable_params:,}",
            f"{total_params:,}",
            100 * trainable_params / total_params,
        )

        self.config.monitor_metric = config.get("monitor_metric", "roc_auc")
        self.config.mode = config.get("mode", "max")
        self.logger.info(
            "Initialized BaselineTrainer with %s for binary classification",
            model.__class__.__name__,
        )

    def compute_loss(self, outputs, labels, is_training=True):
        if not torch.is_tensor(outputs):
            raise ValueError(f"Expected tensor logits from model, got {type(outputs)}")

        logits = outputs
        if labels.dim() > 1:
            labels = labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        if hasattr(self.config, "label_smoothing") and self.config.label_smoothing > 0:
            labels = self.apply_label_smoothing(labels, self.config.label_smoothing)

        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        if hasattr(self, "_batch_count"):
            self._batch_count += 1
        else:
            self._batch_count = 1

        if bce_loss.item() > 5.0 and self._batch_count % 20 == 0:
            self.logger.warning(
                "Large BCE loss detected: %.4f at batch %s. "
                "Check learning rate and data distribution.",
                bce_loss.item(),
                self._batch_count,
            )

        if self._batch_count % 50 == 0:
            self.logger.debug(
                "Batch %s (%s) - BCE loss: %.6f, requires_grad: %s, "
                "logits shape: %s, labels shape: %s",
                self._batch_count,
                "TRAINING" if is_training else "VALIDATION",
                bce_loss.item(),
                bce_loss.requires_grad,
                logits.shape,
                labels.shape,
            )

        return {"total": bce_loss, "bce": bce_loss}

    def compute_metrics(self, outputs, labels):
        metrics = {}

        try:
            if not torch.is_tensor(outputs):
                self.logger.warning("Expected logits tensor, got %s", type(outputs))
                return metrics

            logits = outputs
            if logits.dim() == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits)
            if labels.dim() > 1:
                labels = labels.squeeze()

            probs_cpu = probs.detach().cpu().float().numpy()
            labels_cpu = labels.detach().cpu().float().numpy()

            if len(np.unique(labels_cpu)) > 1:
                roc_auc = roc_auc_score(labels_cpu, probs_cpu)
                auprc = average_precision_score(labels_cpu, probs_cpu)
                metrics["roc_auc"] = roc_auc
                metrics["auprc"] = auprc

                if hasattr(self, "_metrics_log_count"):
                    self._metrics_log_count += 1
                else:
                    self._metrics_log_count = 1

                if self._metrics_log_count % 10 == 0:
                    self.logger.debug(
                        "Baseline metrics - ROC-AUC: %.4f, AUPRC: %.4f",
                        roc_auc,
                        auprc,
                    )

        except Exception as exc:
            self.logger.warning("Failed to compute baseline metrics: %s", exc)

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        save_dir=None,
        plot_curves=False,
        stage=None,
    ):
        self.model.train()
        self.logger.info("SimplePredictor training mode: %s", self.model.training)

        return super().train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=getattr(self.config, "num_epochs", 10),
            monitor_metric=getattr(self.config, "monitor_metric", "auprc"),
            mode=getattr(self.config, "mode", "max"),
            early_stopping_patience=getattr(
                self.config,
                "early_stopping_patience",
                6,
            ),
            save_dir=save_dir,
            plot_curves=plot_curves,
            stage=stage or "baseline",
        )

    def train_fold(
        self,
        fold_number,
        train_loader,
        val_loader,
        save_dir,
        plot_curves=True,
        threshold_method="youden",
        target_recall=0.85,
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.current_fold = fold_number

        self.logger.info("Starting baseline training for fold %s", fold_number)
        train_results = self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            plot_curves=plot_curves,
            stage=f"fold_{fold_number}_baseline",
        )

        val_outputs = train_results["val_outputs"]
        val_labels = train_results["val_labels"]

        if not torch.is_tensor(val_outputs):
            raise ValueError(
                f"Expected logits tensor from model, got {type(val_outputs)}"
            )

        val_logits = val_outputs
        if val_logits.dim() == 2 and val_logits.shape[1] == 1:
            val_logits = val_logits.squeeze(1)

        val_probabilities = torch.sigmoid(val_logits)
        optimal_threshold = find_optimal_threshold(
            predictions=val_probabilities.cpu().float().numpy(),
            labels=val_labels.cpu().float().numpy(),
            method=threshold_method,
            target_recall=target_recall,
        )

        try:
            self.plot_threshold_analysis(
                outputs=val_probabilities.cpu().float().numpy(),
                labels=val_labels.cpu().float().numpy(),
                save_dir=save_dir,
                fold_number=fold_number,
                threshold_methods=[threshold_method],
                target_recalls=[target_recall],
            )
        except Exception as exc:
            self.logger.warning("Failed to plot threshold analysis: %s", exc)

        threshold_metrics = multi_scores(
            y_true=val_labels.cpu().float().numpy(),
            y_pred_proba=val_probabilities.cpu().float().numpy(),
            threshold=optimal_threshold,
        )

        fold_results = {
            "fold": fold_number,
            "epochs_trained": train_results["epochs_trained"],
            "optimal_threshold": optimal_threshold,
            "threshold_method": threshold_method,
            "val_predictions": val_probabilities.cpu().float().numpy(),
            "val_labels": val_labels.cpu().float().numpy(),
            "val_metrics": threshold_metrics,
        }

        if "best_checkpoint_path" in train_results:
            fold_results["best_checkpoint_path"] = str(
                train_results["best_checkpoint_path"]
            )

        self.logger.info("Fold %s baseline completed successfully", fold_number)
        for metric_name, metric_value in threshold_metrics.items():
            if isinstance(metric_value, (int, float)):
                if metric_name in [
                    "roc_auc",
                    "auprc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "mcc",
                ]:
                    self.logger.info("%s: %.4f", metric_name, metric_value)
                else:
                    self.logger.info("%s: %s", metric_name, metric_value)
            else:
                self.logger.info("%s: %s", metric_name, metric_value)

        return fold_results


class PretrainTrainer(BaseTrainer):
    """Trainer for prototype-aware pretraining workflows."""

    def __init__(
        self,
        model,
        config,
        device="cuda",
        save_checkpoints=False,
        logger=None,
    ):
        super().__init__(model, config, device, save_checkpoints, logger)

        self.contrastive_type = getattr(config, "contrastive_type", "prototype")
        self._setup_contrastive_loss()
        self.prototypes = None
        self._configure_parameter_freezing()
        self._always_save_checkpoints = True

        self.logger.info(
            "Initialized PretrainTrainer with contrastive_type='%s'",
            self.contrastive_type,
        )

    def _setup_contrastive_loss(self):
        if self.contrastive_type == "prototype":
            params = getattr(self.config, "prototype_loss_params", {})
            self.logger.info("prototype_loss_params: %s", params)
            self.contrastive_loss_fn = ConSupPrototypeLoss(
                **params,
                device=self.device,
                logger=self.logger,
            )
            self.prototype_update_strategy = params.get(
                "prototype_update_strategy",
                "ema",
            )
            self.prototype_momentum = params.get("prototype_momentum", 0.99)
            self.logger.info("Initialized ConSupPrototypeLoss with params: %s", params)
            self.logger.info(
                "Prototype update strategy: %s", self.prototype_update_strategy
            )
            if self.prototype_update_strategy == "fixed":
                self.logger.info(
                    "Prototypes will be initialized once and then remain fixed."
                )
        elif self.contrastive_type == "hybrid":
            params = getattr(self.config, "hybrid_loss_params", {})
            self.contrastive_loss_fn = HybridContrastiveLoss(
                **params,
                device=self.device,
                logger=self.logger,
            )
            self.prototype_update_strategy = params.get(
                "prototype_update_strategy",
                "ema",
            )
            self.prototype_momentum = params.get("prototype_momentum", 0.99)
            self.logger.info(
                "Initialized HybridContrastiveLoss with params: %s", params
            )
            self.logger.info(
                "Prototype update strategy: %s", self.prototype_update_strategy
            )
            if self.prototype_update_strategy == "fixed":
                self.logger.info(
                    "Prototypes will be initialized once and then remain fixed."
                )
        elif self.contrastive_type == "prototype_ranking":
            params = getattr(self.config, "prototype_loss_params", {})
            self.logger.info("prototype_loss_params: %s", params)

            ranking_loss_type = _config_get(params, "ranking_loss_type")
            if ranking_loss_type is not None and ranking_loss_type != "bce":
                raise ValueError(
                    "PrototypeBCELoss only supports ranking_loss_type='bce' "
                    f"in prototype_ranking configs, got '{ranking_loss_type}'."
                )

            bce_weight = _config_get(
                params,
                "bce_weight",
                _config_get(params, "ranking_weight", 1.0),
            )
            scoring_temperature = _config_get(params, "scoring_temperature", 1.0)
            self.logger.info(
                "Resolved prototype ranking compatibility params: "
                "bce_weight=%s, scoring_temperature=%s",
                bce_weight,
                scoring_temperature,
            )

            self.contrastive_loss_fn = PrototypeBCELoss(
                temperature=_config_get(params, "temperature", 0.07),
                eps_pos=_config_get(params, "eps_pos"),
                eps_neg=_config_get(params, "eps_neg"),
                prototype_weight=_config_get(params, "prototype_weight", 1.0),
                unsupervised_weight=_config_get(params, "unsupervised_weight", 0.5),
                bce_weight=bce_weight,
                scoring_temperature=scoring_temperature,
                prototype_update_strategy=_config_get(
                    params,
                    "prototype_update_strategy",
                    "ema",
                ),
                prototype_momentum=_config_get(params, "prototype_momentum", 0.99),
                device=self.device,
                logger=self.logger,
            )
            self.prototype_update_strategy = _config_get(
                params,
                "prototype_update_strategy",
                "ema",
            )
            self.prototype_momentum = _config_get(params, "prototype_momentum", 0.99)
            self.logger.info("Initialized PrototypeBCELoss with params: %s", params)
            self.logger.info(
                "Prototype update strategy: %s", self.prototype_update_strategy
            )
            if self.prototype_update_strategy == "fixed":
                self.logger.info(
                    "Prototypes will be initialized once and then remain fixed."
                )
        else:
            raise ValueError(f"Unknown contrastive_type: {self.contrastive_type}")

    def _configure_parameter_freezing(self):
        if hasattr(self.model, "set_training_mode"):
            self.model.set_training_mode("pretraining")
            self.logger.info("Set model to pretraining mode (embeddings-only output)")

        has_freeze_encoder = hasattr(self.model, "freeze_encoder")
        has_freeze_classification = hasattr(
            self.model,
            "freeze_classification_head",
        )
        has_freeze_contrastive = hasattr(self.model, "freeze_contrastive_head")

        if has_freeze_encoder and has_freeze_classification:
            self.logger.info(
                "Pretraining: Freezing classification head, keeping encoder and "
                "contrastive head trainable"
            )
            self.model.freeze_classification_head(freeze=True)
            self.model.freeze_encoder(freeze=False)
            if has_freeze_contrastive and getattr(self.model, "use_contrastive", False):
                self.model.freeze_contrastive_head(freeze=False)

            trainable_params = sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            total_params = sum(
                parameter.numel() for parameter in self.model.parameters()
            )
            self.logger.info(
                "After freezing: %s / %s parameters trainable (%.1f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                100 * trainable_params / total_params,
            )
        else:
            self.logger.info(
                "Model does not have freezing methods - keeping all "
                "parameters trainable"
            )

    def _initialize_prototypes(self, train_loader):
        self.logger.info(
            "Gathering input features and labels from all %s batches for "
            "prototype initialization...",
            len(train_loader),
        )

        all_features = []
        all_labels = []

        for batch in train_loader:
            features, labels = self._prepare_batch(batch)
            batch_features = features[0] if isinstance(features, tuple) else features
            batch_features = self._handle_variant_embeddings(
                batch_features, use_mean=True
            )
            all_features.append(batch_features.detach())

            if labels.dim() > 1:
                labels = labels.squeeze()
            all_labels.append(labels.detach())

        if not all_features:
            raise RuntimeError(
                "Could not gather any features for prototype initialization."
            )

        all_features_tensor = torch.cat(all_features, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        unique_labels = torch.unique(all_labels_tensor)
        if len(unique_labels) < 2:
            raise RuntimeError(
                "Prototype initialization requires both classes. "
                f"Found only: {unique_labels.tolist()}"
            )

        majority_class = getattr(self.config, "majority_class", 0)
        minority_class = 1 - majority_class
        mask_minority = all_labels_tensor == minority_class
        mask_majority = all_labels_tensor == majority_class

        n_minority = mask_minority.sum().item()
        n_majority = mask_majority.sum().item()
        if n_minority == 0 or n_majority == 0:
            raise ValueError(
                "Both classes must be present for prototype initialization. "
                f"Minority class ({minority_class}): {n_minority}, "
                f"Majority class ({majority_class}): {n_majority}"
            )

        min_class_size = 5
        if n_minority < min_class_size or n_majority < min_class_size:
            self.logger.warning(
                "Small class sizes detected: Minority class: %s, Majority class: %s. "
                "Minimum recommended: %s per class.",
                n_minority,
                n_majority,
                min_class_size,
            )

        minority_proto = all_features_tensor[mask_minority].mean(dim=0)
        minority_proto = F.normalize(minority_proto, p=2, dim=0)
        majority_proto = -minority_proto

        if minority_class == 0:
            prototypes = torch.stack([minority_proto, majority_proto])
        else:
            prototypes = torch.stack([majority_proto, minority_proto])

        if torch.isnan(prototypes).any() or torch.isinf(prototypes).any():
            raise ValueError("Prototypes contain NaN or Inf values")

        proto_similarity = torch.cosine_similarity(
            prototypes[0],
            prototypes[1],
            dim=0,
        ).item()
        proto_norms = torch.norm(prototypes, p=2, dim=1)

        self.logger.info("Prototype initialization completed:")
        self.logger.info(
            "  - Minority class (%s) samples: %s, Majority class (%s) samples: %s",
            minority_class,
            n_minority,
            majority_class,
            n_majority,
        )
        self.logger.info("  - Focus: Minority class prototype (positive)")
        self.logger.info("  - Prototype similarity: %.4f", proto_similarity)
        self.logger.info(
            "  - Prototype norms: [%.4f, %.4f]",
            proto_norms[0],
            proto_norms[1],
        )

        if proto_similarity > 0.8:
            self.logger.warning(
                "Prototypes are very similar (similarity: %.4f). This may cause "
                "large alignment losses and poor ranking performance.",
                proto_similarity,
            )
        elif proto_similarity < 0.1:
            self.logger.info(
                "Excellent prototype separation achieved (similarity: %.4f)",
                proto_similarity,
            )

        self.prototypes = prototypes
        self._sync_prototypes_to_loss_fn()

        self.logger.info(
            "Prototypes initialized using %s samples from all %s batches",
            all_features_tensor.shape[0],
            len(train_loader),
        )

    def _sync_prototypes_to_loss_fn(self):
        self.logger.debug("_sync_prototypes_to_loss_fn called")
        self.logger.debug("Contrastive type: %s", self.contrastive_type)
        self.logger.debug("Prototypes is None: %s", self.prototypes is None)
        self.logger.debug(
            "Loss function has set_prototypes: %s",
            hasattr(self.contrastive_loss_fn, "set_prototypes"),
        )

        if (
            self.contrastive_type in ["prototype", "hybrid", "prototype_ranking"]
            and self.prototypes is not None
            and hasattr(self.contrastive_loss_fn, "set_prototypes")
        ):
            self.logger.debug(
                "Syncing prototypes to loss function. Shape: %s",
                self.prototypes.shape,
            )

            if self.prototypes.shape[0] == 2:
                proto_similarity = torch.cosine_similarity(
                    self.prototypes[0],
                    self.prototypes[1],
                    dim=0,
                ).item()
                proto_norms = torch.norm(self.prototypes, p=2, dim=1)
                self.logger.debug(
                    "Prototype diagnostics - similarity: %.4f, norms: [%.4f, %.4f]",
                    proto_similarity,
                    proto_norms[0],
                    proto_norms[1],
                )

                if proto_similarity > 0.8:
                    self.logger.warning(
                        "Prototypes are very similar (similarity: %.4f). "
                        "This may cause large alignment losses and poor "
                        "ranking performance.",
                        proto_similarity,
                    )

            try:
                self.contrastive_loss_fn.set_prototypes(self.prototypes)
                if (
                    hasattr(self.contrastive_loss_fn, "prototypes")
                    and self.contrastive_loss_fn.prototypes is not None
                ):
                    self.logger.debug(
                        "Successfully synced prototypes to loss function. "
                        "Loss fn prototypes shape: %s",
                        self.contrastive_loss_fn.prototypes.shape,
                    )
                else:
                    self.logger.error(
                        "Failed to sync prototypes to loss function - prototypes "
                        "may not be used in loss computation"
                    )
                    raise RuntimeError("Prototype synchronization failed")
            except Exception as exc:
                self.logger.error("Error during prototype synchronization: %s", exc)
                raise RuntimeError(f"Prototype synchronization failed: {exc}") from exc
        else:
            self.logger.warning("Skipping prototype sync - conditions not met")

    def _update_prototypes_ema(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = self._handle_variant_embeddings(embeddings, use_mean=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        majority_class = getattr(self.config, "majority_class", 0)
        minority_class = 1 - majority_class

        if self.prototypes is None:
            mask_minority = labels.squeeze() == minority_class
            mask_majority = labels.squeeze() == majority_class

            if mask_minority.any() and mask_majority.any():
                minority_proto = embeddings[mask_minority].mean(dim=0)
                minority_proto = F.normalize(minority_proto, p=2, dim=0)
                majority_proto = -minority_proto
                if minority_class == 0:
                    self.prototypes = torch.stack([minority_proto, majority_proto])
                else:
                    self.prototypes = torch.stack([majority_proto, minority_proto])
            else:
                self.logger.warning(
                    "Batch missing one or both classes - "
                    "skipping prototype initialization"
                )
                return
        else:
            prototype_momentum = self.prototype_momentum
            minority_mask = labels.squeeze() == minority_class

            if minority_mask.any():
                batch_minority_proto = embeddings[minority_mask].mean(dim=0)
                batch_minority_proto = F.normalize(batch_minority_proto, p=2, dim=0)
                current_minority_proto = self.prototypes[minority_class]
                updated_minority_proto = (
                    prototype_momentum * current_minority_proto
                    + (1 - prototype_momentum) * batch_minority_proto
                )
                updated_minority_proto = F.normalize(updated_minority_proto, p=2, dim=0)
                updated_majority_proto = -updated_minority_proto

                if minority_class == 0:
                    self.prototypes = torch.stack(
                        [updated_minority_proto, updated_majority_proto]
                    )
                else:
                    self.prototypes = torch.stack(
                        [updated_majority_proto, updated_minority_proto]
                    )

                if hasattr(self, "device"):
                    self.prototypes = self.prototypes.to(device=self.device)

                self.logger.debug(
                    "Updated prototypes - minority class %s, momentum: %.4f, "
                    "batch samples: %s",
                    minority_class,
                    prototype_momentum,
                    minority_mask.sum().item(),
                )
            else:
                self.logger.warning(
                    "No minority class samples in batch - skipping prototype update"
                )

        self._sync_prototypes_to_loss_fn()

    def on_batch_end(self, outputs, labels, batch_idx):
        if self.contrastive_type in ["prototype", "hybrid", "prototype_ranking"]:
            prototype_update_strategy = self.prototype_update_strategy

            if prototype_update_strategy == "warmup_freeze":
                warmup_epochs = getattr(self.config, "warmup_epochs", 5)
                effective_strategy = (
                    "ema" if self.current_epoch < warmup_epochs else "fixed"
                )
            else:
                effective_strategy = prototype_update_strategy

            if effective_strategy == "ema":
                if torch.is_tensor(outputs):
                    self._update_prototypes_ema(outputs.detach(), labels)
                else:
                    self.logger.error(
                        "Expected tensor outputs for prototype updates, got %s",
                        type(outputs),
                    )

    def on_epoch_start(self, epoch, train_loader):
        if self.contrastive_type in ["prototype", "hybrid", "prototype_ranking"]:
            prototype_update_strategy = self.prototype_update_strategy
            if prototype_update_strategy == "warmup_freeze":
                warmup_epochs = getattr(self.config, "warmup_epochs", 5)
                if epoch == warmup_epochs:
                    self.logger.info(
                        "Hybrid prototype strategy: Warmup phase ended at epoch %s. "
                        "Prototypes will now be frozen for remaining training.",
                        epoch,
                    )

    def _get_extra_checkpoint_state(self):
        extra_state = {}
        if self.prototypes is not None:
            extra_state["prototypes"] = self.prototypes.cpu()
            self.logger.debug("Saving prototypes with shape: %s", self.prototypes.shape)
        return extra_state

    def _prepare_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            features, labels = batch[0], batch[1]
            if isinstance(features, (tuple, list)):
                features = tuple(feature.to(self.device) for feature in features)
            else:
                features = features.to(self.device)
            labels = labels.to(self.device)
            return features, labels
        raise ValueError("Batch format not recognized in pretraining")

    def compute_loss(self, outputs, labels, is_training=True):
        if not torch.is_tensor(outputs):
            raise ValueError("Model must return embeddings tensor for pretraining")

        embeddings = outputs
        if labels.dim() > 1:
            labels = labels.squeeze()
        if labels.dim() == 0:
            self.logger.warning(
                "Detected scalar label tensor (batch size 1). Converting to 1D tensor. "
                "Original labels shape: %s",
                labels.shape,
            )
            labels = labels.unsqueeze(0)

        one_hot_labels = F.one_hot(labels.long(), num_classes=2).float()

        if hasattr(self, "_batch_count"):
            self._batch_count += 1
        else:
            self._batch_count = 1

        if self._batch_count % 100 == 0:
            self.logger.debug(
                "Batch %s: embeddings shape: %s, original labels shape: %s, "
                "one_hot_labels shape: %s",
                self._batch_count,
                embeddings.shape,
                labels.shape,
                one_hot_labels.shape,
            )

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)

        loss_components = self.contrastive_loss_fn(
            features=embeddings,
            labels=one_hot_labels,
        )
        return loss_components

    def compute_metrics(self, outputs, labels):
        return {}

    def train(
        self,
        train_loader,
        val_loader,
        save_dir=None,
        plot_curves=False,
        stage=None,
    ):
        if (
            self.contrastive_type in ["prototype", "hybrid", "prototype_ranking"]
            and self.prototypes is None
        ):
            self.logger.info("Initializing prototypes before training...")
            self._initialize_prototypes(train_loader)

        monitor_metric = getattr(self.config, "monitor_metric", "loss")
        mode = getattr(
            self.config,
            "mode",
            "min" if monitor_metric == "loss" else "max",
        )

        return super().train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.num_epochs,
            monitor_metric=monitor_metric,
            mode=mode,
            early_stopping_patience=getattr(
                self.config,
                "early_stopping_patience",
                10,
            ),
            save_dir=save_dir,
            plot_curves=plot_curves,
            stage=stage or "pretraining",
        )


class PrototypeRankingTrainer(PretrainTrainer):
    """Trainer for prototype-based learning with optional BCE classification."""

    def __init__(
        self,
        model,
        config,
        device="cuda",
        save_checkpoints=False,
        logger=None,
    ):
        if not hasattr(config, "contrastive_type") or config.contrastive_type not in [
            "prototype_ranking",
            "hybrid",
        ]:
            raise ValueError(
                "PrototypeRankingTrainer requires contrastive_type='prototype_ranking' "
                "or 'hybrid'"
            )

        super().__init__(
            model=model,
            config=config,
            device=device,
            save_checkpoints=save_checkpoints,
            logger=logger,
        )

        expected_loss_types = [PrototypeBCELoss, HybridContrastiveLoss]
        if not any(
            isinstance(self.contrastive_loss_fn, loss_type)
            for loss_type in expected_loss_types
        ):
            raise ValueError(
                "Expected PrototypeBCELoss or HybridContrastiveLoss, got "
                f"{type(self.contrastive_loss_fn)}"
            )

        self.config.monitor_metric = config.get("monitor_metric", "auprc")
        self.config.mode = config.get("mode", "max")
        self.best_prototypes = None
        self.best_epoch = -1
        self.model.train()

        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        self.logger.info(
            "Model: %s / %s parameters trainable (%.1f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            100 * trainable_params / total_params,
        )

        prototype_strategy = getattr(self.config, "prototype_update_strategy", "ema")
        if prototype_strategy == "warmup_freeze":
            warmup_epochs = getattr(self.config, "warmup_epochs", 5)
            self.logger.info(
                "Using hybrid prototype strategy: %s epochs warmup + "
                "freeze for ranking",
                warmup_epochs,
            )
        else:
            self.logger.info("Using prototype strategy: %s", prototype_strategy)

        if isinstance(self.contrastive_loss_fn, HybridContrastiveLoss):
            self.logger.info(
                "Initialized PrototypeRankingTrainer for hybrid contrastive learning"
            )
        else:
            self.logger.info(
                "Initialized PrototypeRankingTrainer for prototype ranking"
            )

    def load_pretrained_model_and_prototypes(
        self,
        run_dir: Path,
        fold_number: int,
    ):
        try:
            checkpoint_path = (
                run_dir / f"fold_{fold_number}" / "pretraining" / "checkpoint.pt"
            )
            if not checkpoint_path.exists():
                self.logger.warning(
                    "Pretrained checkpoint not found at %s", checkpoint_path
                )
                return False, None

            self.logger.info("Loading pretrained model from %s", checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.logger.info("Successfully loaded pretrained model state")

            if "prototypes" in checkpoint and checkpoint["prototypes"] is not None:
                self.prototypes = checkpoint["prototypes"].to(self.device)
                self._sync_prototypes_to_loss_fn()
                self.logger.info(
                    "Loaded pretrained prototypes: %s", self.prototypes.shape
                )
            elif (
                "extra_state" in checkpoint
                and "prototypes" in checkpoint["extra_state"]
            ):
                self.prototypes = checkpoint["extra_state"]["prototypes"].to(
                    self.device
                )
                self._sync_prototypes_to_loss_fn()
                self.logger.info(
                    "Loaded pretrained prototypes from extra_state: %s",
                    self.prototypes.shape,
                )
            else:
                self.logger.warning(
                    "No prototypes found in checkpoint - will initialize from scratch"
                )

            trainable_params = sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            total_params = sum(
                parameter.numel() for parameter in self.model.parameters()
            )
            self.logger.info(
                "Loaded model: %s / %s parameters trainable (%.1f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                100 * trainable_params / total_params,
            )

            return True, str(checkpoint_path)

        except Exception as exc:
            self.logger.error("Failed to load pretrained model: %s", exc)
            return False, None

    def compute_loss(self, outputs, labels, is_training=True):
        if not torch.is_tensor(outputs):
            raise ValueError(
                f"Expected tensor embeddings from model, got {type(outputs)}"
            )

        embeddings = outputs
        if labels.dim() > 1:
            labels = labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        one_hot_labels = F.one_hot(labels.long(), num_classes=2).float()

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
            self.logger.debug(
                "No variants detected, reshaped embeddings to %s", embeddings.shape
            )
        elif embeddings.dim() == 3:
            self.logger.debug(
                "Variants detected, embeddings shape: %s", embeddings.shape
            )
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        if is_training and not embeddings.requires_grad:
            self.logger.error(
                "Embeddings do not require gradients during training! Shape: %s",
                embeddings.shape,
            )
        elif not is_training and not embeddings.requires_grad:
            self.logger.debug(
                "Validation mode: embeddings gradients disabled (expected). Shape: %s",
                embeddings.shape,
            )

        loss_components = self.contrastive_loss_fn(
            features=embeddings,
            labels=one_hot_labels,
        )

        if hasattr(self, "_batch_count"):
            self._batch_count += 1
        else:
            self._batch_count = 1

        if "prototype" in loss_components:
            proto_loss_val = (
                loss_components["prototype"].item()
                if torch.is_tensor(loss_components["prototype"])
                else loss_components["prototype"]
            )
            if proto_loss_val > 5.0 and self._batch_count % 20 == 0:
                self.logger.warning(
                    "Large prototype loss detected: %.4f at batch %s. "
                    "Check prototype initialization and temperature settings.",
                    proto_loss_val,
                    self._batch_count,
                )

        if "total" in loss_components:
            total_loss = loss_components["total"]
            if is_training and not total_loss.requires_grad:
                self.logger.error(
                    "Total loss does not require gradients during training "
                    "at batch %s!",
                    self._batch_count,
                )
            elif not is_training:
                self.logger.debug(
                    "Validation mode: total loss gradients disabled (expected) "
                    "at batch %s",
                    self._batch_count,
                )

            if self._batch_count % 50 == 0:
                self.logger.debug(
                    "Batch %s (%s) - Total loss: %.6f, requires_grad: %s, "
                    "embeddings shape: %s, embeddings requires_grad: %s",
                    self._batch_count,
                    "TRAINING" if is_training else "VALIDATION",
                    total_loss.item(),
                    total_loss.requires_grad,
                    embeddings.shape,
                    embeddings.requires_grad,
                )

        return loss_components

    def on_epoch_end(self, epoch, val_metrics, is_best=False):
        if is_best and self.prototypes is not None:
            self.best_prototypes = self.prototypes.clone().detach()
            self.best_epoch = epoch
            self.logger.info("Saved best prototypes from epoch %s", epoch)

    def compute_metrics(self, outputs, labels):
        metrics = {}

        try:
            if not torch.is_tensor(outputs):
                self.logger.warning("Expected embeddings tensor, got %s", type(outputs))
                return metrics

            embeddings = outputs
            if self.prototypes is None:
                self.logger.warning(
                    "Cannot compute metrics: prototypes not initialized"
                )
                return metrics

            embeddings = embeddings.float()
            prototypes = self.prototypes.float()

            if embeddings.dim() == 3:
                self.logger.debug(
                    "Using all %s variants for evaluation (consistent with training)",
                    embeddings.shape[1],
                )
            elif embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)
                self.logger.debug(
                    "Single variant detected, added dimension for consistency"
                )
            else:
                raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

            scoring_temperature = getattr(
                self.contrastive_loss_fn,
                "scoring_temperature",
                1.0,
            )
            ranking_scores = compute_prototype_distance_scores(
                embeddings=embeddings,
                prototypes=prototypes,
                scoring_temperature=scoring_temperature,
                logger=self.logger,
            )
            probs = torch.sigmoid(ranking_scores)

            if labels.dim() > 1:
                labels = labels.squeeze()

            probs_cpu = probs.detach().cpu().float().numpy()
            labels_cpu = labels.detach().cpu().float().numpy()

            if len(np.unique(labels_cpu)) > 1:
                roc_auc = roc_auc_score(labels_cpu, probs_cpu)
                auprc = average_precision_score(labels_cpu, probs_cpu)
                metrics["roc_auc"] = roc_auc
                metrics["auprc"] = auprc

                if hasattr(self, "_metrics_log_count"):
                    self._metrics_log_count += 1
                else:
                    self._metrics_log_count = 1

                if self._metrics_log_count % 10 == 0:
                    self.logger.debug(
                        "Prototype ranking metrics - ROC-AUC: %.4f, AUPRC: %.4f",
                        roc_auc,
                        auprc,
                    )

        except Exception as exc:
            self.logger.warning("Failed to compute prototype ranking metrics: %s", exc)

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        save_dir=None,
        plot_curves=False,
        stage=None,
        pretrained_run_dir: Optional[Path] = None,
        fold_number: Optional[int] = None,
    ):
        self.model.train()

        if pretrained_run_dir is not None and fold_number is not None:
            self.logger.info(
                "Attempting to load pretrained model from %s", pretrained_run_dir
            )
            success, checkpoint_path = self.load_pretrained_model_and_prototypes(
                pretrained_run_dir,
                fold_number,
            )
            if success:
                self.logger.info(
                    "Successfully loaded pretrained model from %s", checkpoint_path
                )
                if self.prototypes is not None:
                    self.logger.info("Using loaded pretrained prototypes")
                else:
                    self.logger.warning(
                        "No prototypes loaded - will initialize from scratch"
                    )
                    self._initialize_prototypes(train_loader)
            else:
                self.logger.warning(
                    "Failed to load pretrained model - initializing from scratch"
                )
                self._initialize_prototypes(train_loader)
        else:
            if self.prototypes is None:
                self.logger.info("Initializing prototypes before training...")
                self._initialize_prototypes(train_loader)
            else:
                self.logger.info(
                    "Prototypes already loaded - ensuring synchronization with "
                    "loss function"
                )
                self._sync_prototypes_to_loss_fn()

        self.logger.info("Model training mode: %s", self.model.training)

        if hasattr(self.contrastive_loss_fn, "prototypes"):
            if self.contrastive_loss_fn.prototypes is not None:
                self.logger.info(
                    "Loss function prototypes shape: %s",
                    self.contrastive_loss_fn.prototypes.shape,
                )
                self.logger.info(
                    "Loss function prototypes device: %s",
                    self.contrastive_loss_fn.prototypes.device,
                )
            else:
                self.logger.error("Prototypes not set in loss function!")
                raise RuntimeError(
                    "Prototypes not properly synchronized with loss function"
                )
        else:
            self.logger.error("Loss function does not have prototypes attribute!")
            raise RuntimeError("Loss function does not support prototypes")

        return super().train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            plot_curves=plot_curves,
            stage=stage or "prototype_ranking",
        )

    def train_fold(
        self,
        fold_number,
        train_loader,
        val_loader,
        save_dir,
        plot_curves=True,
        threshold_method=None,
        target_recall=0.85,
        pretrained_run_dir: Optional[Path] = None,
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.current_fold = fold_number

        self.logger.info("Starting prototype ranking training for fold %s", fold_number)

        if threshold_method is None:
            self.logger.warning(
                "threshold_method is None - this should not happen in normal operation"
            )
            threshold_method = "youden"
        else:
            self.logger.info("Using threshold method: %s", threshold_method)

        train_results = self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            plot_curves=plot_curves,
            stage=f"fold_{fold_number}_prototype_ranking",
            pretrained_run_dir=pretrained_run_dir,
            fold_number=fold_number,
        )

        val_outputs = train_results["val_outputs"]
        val_labels = train_results["val_labels"]

        if self.best_prototypes is not None:
            self.logger.info(
                "Using best prototypes from epoch %s for final evaluation",
                self.best_epoch,
            )
            self.prototypes = self.best_prototypes.to(self.device)
            self._sync_prototypes_to_loss_fn()

            if self.prototypes.shape[0] == 2:
                proto_similarity = torch.cosine_similarity(
                    self.prototypes[0],
                    self.prototypes[1],
                    dim=0,
                ).item()
                proto_norms = torch.norm(self.prototypes, p=2, dim=1)
                self.logger.info(
                    "Best prototypes diagnostics - similarity: %.4f, norms: "
                    "[%.4f, %.4f]",
                    proto_similarity,
                    proto_norms[0],
                    proto_norms[1],
                )
        else:
            self.logger.warning(
                "No best prototypes available, using current prototypes"
            )

        if not torch.is_tensor(val_outputs):
            raise ValueError(f"Expected embeddings tensor, got {type(val_outputs)}")

        embeddings = val_outputs.float()
        prototypes = self.prototypes.float()
        scoring_temperature = getattr(
            self.contrastive_loss_fn,
            "scoring_temperature",
            1.0,
        )
        ranking_scores = compute_prototype_distance_scores(
            embeddings=embeddings,
            prototypes=prototypes,
            scoring_temperature=scoring_temperature,
            logger=self.logger,
        )
        val_probabilities = torch.sigmoid(ranking_scores)

        if embeddings.dim() == 3:
            self.logger.debug(
                "Using all variants for final evaluation, averaged %s variants",
                embeddings.shape[1],
            )

        self.logger.info(
            "Calling find_optimal_threshold with method: '%s', target_recall: %s",
            threshold_method,
            target_recall,
        )
        optimal_threshold = find_optimal_threshold(
            predictions=val_probabilities.cpu().float().numpy(),
            labels=val_labels.cpu().float().numpy(),
            method=threshold_method,
            target_recall=target_recall,
        )
        self.logger.info(
            "find_optimal_threshold returned optimal_threshold: %s",
            optimal_threshold,
        )

        try:
            self.plot_threshold_analysis(
                outputs=val_probabilities.cpu().float().numpy(),
                labels=val_labels.cpu().float().numpy(),
                save_dir=save_dir,
                fold_number=fold_number,
                threshold_methods=[threshold_method],
                target_recalls=[target_recall],
            )
        except Exception as exc:
            self.logger.warning("Failed to plot threshold analysis: %s", exc)

        self.logger.info(
            "Computing metrics with optimal threshold: %s", optimal_threshold
        )
        threshold_metrics = multi_scores(
            y_true=val_labels.cpu().float().numpy(),
            y_pred_proba=val_probabilities.cpu().float().numpy(),
            threshold=optimal_threshold,
        )

        fold_results = {
            "fold": fold_number,
            "epochs_trained": train_results["epochs_trained"],
            "optimal_threshold": optimal_threshold,
            "threshold_method": threshold_method,
            "val_predictions": val_probabilities.cpu().float().numpy(),
            "val_labels": val_labels.cpu().float().numpy(),
            "val_metrics": threshold_metrics,
        }

        if "best_checkpoint_path" in train_results:
            fold_results["best_checkpoint_path"] = str(
                train_results["best_checkpoint_path"]
            )

        self.logger.info(
            "Fold %s prototype ranking completed successfully", fold_number
        )
        for metric_name, metric_value in threshold_metrics.items():
            if isinstance(metric_value, (int, float)):
                if metric_name in [
                    "roc_auc",
                    "auprc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "mcc",
                ]:
                    self.logger.info("%s: %.4f", metric_name, metric_value)
                else:
                    self.logger.info("%s: %s", metric_name, metric_value)
            else:
                self.logger.info("%s: %s", metric_name, metric_value)

        return fold_results


__all__ = [
    "BaselineTrainer",
    "PretrainTrainer",
    "PrototypeRankingTrainer",
]
