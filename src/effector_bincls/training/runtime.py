"""Package-owned shared training runtime and trainer base classes."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class MetricsTracker:
    """Tracks and stores training and validation metrics."""

    def __init__(self) -> None:
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []

    def update_train(self, metrics: dict[str, Any]) -> None:
        self.train_metrics.append(metrics)

    def update_val(self, metrics: dict[str, Any]) -> None:
        self.val_metrics.append(metrics)


class EarlyStopping:
    """Stop training when the monitored validation metric stops improving."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        logger: logging.Logger | None = None,
        monitor: str = "auc",
        mode: str | None = None,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.monitor = monitor
        self.improved = False
        self.mode = "min" if (mode is None and monitor == "loss") else mode or "max"
        self.best_metric_value = np.inf if self.mode == "min" else -np.inf
        self.delta = delta
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, val_metrics: dict[str, Any] | float | int) -> bool:
        if isinstance(val_metrics, (float, int)):
            metric_value = float(val_metrics)
            metric_name = "loss"
        else:
            metric_name = self.monitor
            if metric_name not in val_metrics:
                raise ValueError(
                    f"Metric '{metric_name}' not found in validation metrics"
                )
            metric_value = float(val_metrics[metric_name])

        score = -metric_value if self.mode == "min" else metric_value
        self.improved = False

        if self.best_score is None:
            self.best_score = score
            self.improved = True
            self._update_best_value(metric_value, metric_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    "EarlyStopping counter: %s out of %s",
                    self.counter,
                    self.patience,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.improved = True
            self._update_best_value(metric_value, metric_name)
            self.counter = 0

        return self.improved

    def _update_best_value(self, metric_value: float, metric_name: str) -> None:
        if self.verbose:
            direction = "decreased" if self.mode == "min" else "increased"
            self.logger.info(
                "Validation %s %s (%.4f --> %.4f)",
                metric_name,
                direction,
                self.best_metric_value,
                metric_value,
            )
        self.best_metric_value = metric_value


class WarmupPlateauScheduler:
    """Combine linear warmup with plateau or cosine scheduling."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_epochs: int,
        scheduler_type: str = "plateau",
        warmup_lr_init: float | None = None,
        mode: str = "min",
        min_lr: float = 1e-6,
        factor: float = 0.7,
        patience: int = 5,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 2,
        total_epochs: int | None = None,
        eta_min: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.optimizer = optimizer
        self.num_warmup_epochs = int(num_warmup_epochs)
        self.scheduler_type = scheduler_type.lower() if scheduler_type else "plateau"
        self.finished_warmup = False
        self.current_epoch = 0

        if self.scheduler_type not in ["plateau", "cosine"]:
            raise ValueError("scheduler_type must be 'plateau' or 'cosine'")

        self.target_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_lr_init = (
            float(warmup_lr_init)
            if warmup_lr_init is not None
            else self.target_lrs[0] * 0.1
        )

        for group in optimizer.param_groups:
            group["lr"] = self.warmup_lr_init

        if self.scheduler_type == "plateau":
            plateau_kwargs = {
                "mode": str(mode),
                "factor": float(factor),
                "patience": int(patience),
                "threshold": float(threshold),
                "threshold_mode": str(threshold_mode),
                "cooldown": int(cooldown),
                "min_lr": float(min_lr),
            }
            if plateau_kwargs["threshold_mode"] not in ["rel", "abs"]:
                raise ValueError("threshold_mode must be 'rel' or 'abs'")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **plateau_kwargs
            )
            self.scheduler_kwargs = plateau_kwargs
        else:
            if total_epochs is None:
                raise ValueError("total_epochs must be specified for cosine scheduler")
            cosine_kwargs = {
                "T_max": int(max(total_epochs - self.num_warmup_epochs, 1)),
                "eta_min": float(eta_min),
            }
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **cosine_kwargs
            )
            self.scheduler_kwargs = cosine_kwargs

    def state_dict(self) -> dict[str, Any]:
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler"]
        }
        state_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        scheduler_state = state_dict.pop("scheduler_state_dict")
        self.__dict__.update(state_dict)
        if self.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.scheduler_kwargs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.scheduler_kwargs
            )
        self.scheduler.load_state_dict(scheduler_state)

    def step(self, metrics: float | int | None = None) -> None:
        if not self.finished_warmup:
            self.current_epoch += 1
            if self.current_epoch >= self.num_warmup_epochs:
                self.finished_warmup = True
                for index, group in enumerate(self.optimizer.param_groups):
                    group["lr"] = self.target_lrs[index]
                print(
                    f"Finished warmup phase ({self.scheduler_type}). Current LR: "
                    f"{self.get_last_lr()[0]:.6f}"
                )
            else:
                warmup_factor = float(self.current_epoch) / float(
                    self.num_warmup_epochs
                )
                for index, group in enumerate(self.optimizer.param_groups):
                    group["lr"] = self.warmup_lr_init + warmup_factor * (
                        self.target_lrs[index] - self.warmup_lr_init
                    )
                print(
                    f"Warmup epoch {self.current_epoch}/{self.num_warmup_epochs}. "
                    f"Current LR: {self.get_last_lr()[0]:.6f}"
                )
                return

        prev_lr = self.get_last_lr()[0]
        if self.scheduler_type == "plateau":
            if metrics is not None:
                self.scheduler.step(float(metrics))
        else:
            self.scheduler.step()
        curr_lr = self.get_last_lr()[0]
        if abs(prev_lr - curr_lr) > 1e-8:
            print(f"Learning rate changed: {prev_lr:.6f} -> {curr_lr:.6f}")

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]


def create_optimizer(
    model: torch.nn.Module,
    config: Any,
    logger: logging.Logger | None = None,
) -> torch.optim.Optimizer:
    """Create the AdamW optimizer with decayed and non-decayed parameter groups."""
    if logger is None:
        logger = logging.getLogger(__name__)

    trainable_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    decay_params = [parameter for parameter in trainable_params if parameter.dim() >= 2]
    nodecay_params = [
        parameter for parameter in trainable_params if parameter.dim() < 2
    ]

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(parameter.numel() for parameter in decay_params)
    num_nodecay_params = sum(parameter.numel() for parameter in nodecay_params)
    logger.info("\nOptimizer configuration:")
    logger.info(
        "  - Creating optimizer for %s trainable parameters.",
        f"{num_decay_params + num_nodecay_params:,}",
    )
    logger.info(
        "  - Num decayed parameter tensors: %s, with %s parameters",
        len(decay_params),
        f"{num_decay_params:,}",
    )
    logger.info(
        "  - Num non-decayed parameter tensors: %s, with %s parameters",
        len(nodecay_params),
        f"{num_nodecay_params:,}",
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = {"fused": True} if use_fused else {}
    logger.info("  - Using fused AdamW: %s", use_fused)
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, **extra_args)


class BaseTrainer(ABC):
    """Simplified shared trainer that implements the package training loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        device: str = "cuda",
        save_checkpoints: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.current_epoch = 0
        self.current_fold: int | None = None
        self.current_stage: str | None = None
        self.phase = "train"
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion = None
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.dtype = torch.bfloat16
        self.logger.info("Initialized mixed precision training with %s", self.dtype)

    def _handle_variant_embeddings(
        self,
        embeddings: torch.Tensor,
        use_mean: bool = True,
    ) -> torch.Tensor:
        if embeddings.dim() == 3:
            return embeddings.mean(dim=1) if use_mean else embeddings[:, 0, :]
        if embeddings.dim() == 2:
            return embeddings
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    def _log_metrics(
        self,
        metrics: dict[str, Any],
        phase: str | None = None,
        step: int | None = None,
        prefix: str = "",
        verbose_logging: bool = False,
    ) -> None:
        phase = phase or self.phase
        step = step if step is not None else self.current_epoch

        def log_metric_dict(
            metrics_dict: dict[str, Any],
            current_prefix: str = "",
        ) -> None:
            for name, value in metrics_dict.items():
                metric_name = f"{current_prefix}{name}" if current_prefix else name
                if (
                    not verbose_logging
                    and phase in ["train", "val"]
                    and any(
                        label_metric in name
                        for label_metric in [
                            "_f1",
                            "_precision",
                            "_recall",
                            "_accuracy",
                            "_mcc",
                            "_auroc",
                            "_auprc",
                            "_jaccard",
                        ]
                    )
                    and not any(
                        aggregate_prefix in name
                        for aggregate_prefix in [
                            "micro_",
                            "macro_",
                            "samples_",
                            "binary_",
                        ]
                    )
                ):
                    continue

                if isinstance(value, dict):
                    log_metric_dict(value, f"{metric_name}_")
                elif isinstance(value, (int, float)):
                    if torch.is_tensor(value):
                        value = value.item()

                    if verbose_logging or phase == "test":
                        should_log = True
                    elif phase in ["train", "val"]:
                        is_loss_or_grad = "loss" in metric_name or metric_name in [
                            "grad_norm",
                            "roc_auc",
                            "auprc",
                        ]
                        is_multilabel_threshold_independent = any(
                            metric_prefix in metric_name
                            for metric_prefix in ["micro_", "macro_"]
                        ) and any(
                            suffix in metric_name
                            for suffix in ["auprc", "auroc", "roc_auc"]
                        )
                        is_binary_threshold_independent = metric_name.startswith(
                            "binary_"
                        ) and any(
                            suffix in metric_name for suffix in ["auroc", "auprc"]
                        )
                        should_log = (
                            is_loss_or_grad
                            or is_multilabel_threshold_independent
                            or is_binary_threshold_independent
                        )
                    else:
                        should_log = False

                    if should_log:
                        self.logger.info("%s: %.3f", metric_name, value)

        if phase in ["train", "val", "test"]:
            self.logger.info("\n%s Metrics:", phase.capitalize())

        log_metric_dict(metrics)

        if phase == "train" and self.optimizer is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info("\nLearning rate: %.2e", lr)

        self.phase = "train"

    def _log_batch_metrics(
        self,
        metrics: dict[str, float],
        phase: str,
        batch_idx: int,
        total_batches: int,
    ) -> None:
        metrics_str = ", ".join(
            [f"{key}: {value:.3f}" for key, value in metrics.items()]
        )
        self.logger.debug(
            "%s - Batch [%s/%s] - %s",
            phase.capitalize(),
            batch_idx + 1,
            total_batches,
            metrics_str,
        )

    def _log_epoch_metrics(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        self.logger.info(
            "\nEpoch [%s/%s]", self.current_epoch + 1, self.config.num_epochs
        )
        self.phase = "train"
        self._log_metrics(train_metrics)
        if val_metrics:
            self.phase = "val"
            self._log_metrics(val_metrics)
        self.phase = "train"

    def save_checkpoint(
        self,
        path: Path,
        is_best: bool = False,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "config": self.config,
        }
        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)
        self.logger.info("Saved checkpoint to %s", path)

        if is_best:
            best_path = path.parent / "best_model.pt"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(path.resolve())

    def load_checkpoint(
        self,
        path: Path,
        load_optimizer: bool = False,
        extra_keys: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.logger.info("Loaded model state from %s", path)

        if (
            load_optimizer
            and self.optimizer is not None
            and "optimizer_state" in checkpoint
            and checkpoint["optimizer_state"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.logger.info("Loaded optimizer state from checkpoint")

        extra_state: dict[str, Any] = {}
        if extra_keys:
            for key in extra_keys:
                if key in checkpoint:
                    extra_state[key] = checkpoint[key]

        return checkpoint, extra_state

    def _move_to_device(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, (tuple, list)):
            return type(data)(self._move_to_device(item) for item in data)
        if hasattr(data, "to"):
            return data.to(self.device)
        return data

    def _concatenate_outputs(self, all_outputs: list[Any]) -> Any:
        if not all_outputs:
            return None
        first_output = all_outputs[0]
        if torch.is_tensor(first_output):
            return torch.cat(all_outputs, dim=0)
        if isinstance(first_output, (tuple, list)):
            concatenated_components = []
            for index in range(len(first_output)):
                component_list = [output[index] for output in all_outputs]
                concatenated_components.append(torch.cat(component_list, dim=0))
            return type(first_output)(concatenated_components)
        self.logger.warning(
            "Unexpected output type in concatenation: %s", type(first_output)
        )
        return all_outputs

    def _concatenate_labels(self, all_labels: list[Any]) -> Any:
        if not all_labels:
            self.logger.warning(
                "_concatenate_labels received empty all_labels list, returning None"
            )
            return None

        first_label = all_labels[0]
        if torch.is_tensor(first_label):
            return torch.cat(all_labels, dim=0)
        if isinstance(first_label, (tuple, list)):
            concatenated_components = []
            for index in range(len(first_label)):
                component_list = [label[index] for label in all_labels]
                concatenated_components.append(torch.cat(component_list, dim=0))
            return tuple(concatenated_components)
        self.logger.warning(
            "Unexpected label type in concatenation: %s", type(first_label)
        )
        return all_labels

    def _prepare_batch(self, batch: Any) -> tuple[Any, Any]:
        if not isinstance(batch, (tuple, list)):
            raise ValueError("Batch format not recognized")

        if len(batch) == 2:
            features, labels = batch
            if isinstance(features, (tuple, list)):
                features = tuple(feature.to(self.device) for feature in features)
            else:
                features = features.to(self.device)
        elif len(batch) == 3:
            feature1, feature2, labels = batch
            features = (feature1.to(self.device), feature2.to(self.device))
        elif len(batch) == 5:
            p1_seq, p1_struct, p2_seq, p2_struct, labels = batch
            features = (
                p1_seq.to(self.device),
                p1_struct.to(self.device),
                p2_seq.to(self.device),
                p2_struct.to(self.device),
            )
        else:
            raise ValueError(f"Unsupported batch format with {len(batch)} elements")

        labels = self._move_to_device(labels)
        return features, labels

    @abstractmethod
    def compute_loss(
        self,
        outputs: Any,
        labels: Any,
        is_training: bool = True,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, outputs: Any, labels: Any) -> dict[str, Any]:
        raise NotImplementedError

    def apply_label_smoothing(
        self,
        labels: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        if alpha is None:
            alpha = getattr(self.config, "label_smoothing", 0.0)
        if alpha <= 0:
            return labels
        return labels * (1 - alpha) + (1 - labels) * alpha

    def train_epoch(
        self,
        train_loader: Any,
        metrics_tracker: MetricsTracker,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_components: dict[str, float] = {}
        all_outputs: list[Any] = []
        all_labels: list[Any] = []
        total_grad_norm = 0.0
        n_batches = len(train_loader)

        if n_batches == 0:
            self.logger.warning(
                "train_loader is empty, skipping training for this epoch."
            )
            return {"loss": 0.0, "grad_norm": 0.0}

        for batch_idx, batch in enumerate(train_loader):
            try:
                features, labels = self._prepare_batch(batch)
                assert self.optimizer is not None
                self.optimizer.zero_grad()

                device_type = "cuda" if "cuda" in self.device else "cpu"
                with torch.autocast(device_type=device_type, dtype=self.dtype):
                    outputs = (
                        self.model(*features)
                        if isinstance(features, tuple)
                        else self.model(features)
                    )
                    logits_to_check = (
                        outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    )
                    if torch.isnan(logits_to_check).any():
                        raise RuntimeError(
                            f"NaN detected in model output logits at batch {batch_idx}"
                        )
                    loss_output = self.compute_loss(outputs, labels, is_training=True)

                if isinstance(loss_output, dict):
                    if "total" not in loss_output:
                        raise ValueError(
                            "When returning a dictionary of losses, "
                            "'total' must be included"
                        )
                    total_loss_batch = loss_output["total"]
                    if not loss_components:
                        loss_components = {key: 0.0 for key in loss_output.keys()}
                    for name, component_loss in loss_output.items():
                        loss_components[name] += component_loss.item()
                else:
                    total_loss_batch = loss_output
                    total_loss += total_loss_batch.item()

                if torch.isnan(total_loss_batch).any():
                    raise RuntimeError(f"NaN loss detected at batch {batch_idx}")

                total_loss_batch.backward()

                max_norm = getattr(self.config, "grad_clip_value", float("inf"))
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm,
                )
                if torch.isnan(grad_norm):
                    raise RuntimeError(f"NaN gradients detected at batch {batch_idx}")

                total_grad_norm += grad_norm.item()
                self.optimizer.step()

                if hasattr(self, "on_batch_end"):
                    self.on_batch_end(outputs, labels, batch_idx)

                if torch.is_tensor(outputs):
                    all_outputs.append(outputs.detach())
                elif isinstance(outputs, (list, tuple)):
                    all_outputs.append(outputs)
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
                all_labels.append(labels)

            except RuntimeError as exc:
                if "out of memory" in str(exc):
                    self.logger.error(
                        "CUDA out of memory at batch %s. "
                        "Try reducing batch size or model size.",
                        batch_idx,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise
            except Exception as exc:
                self.logger.error("Error in batch %s: %s", batch_idx, exc)
                raise

        if loss_components:
            metrics = {
                f"loss_{name}": total / n_batches
                for name, total in loss_components.items()
                if name != "total"
            }
            metrics["loss"] = loss_components["total"] / n_batches
        else:
            metrics = {"loss": total_loss / n_batches}

        metrics["grad_norm"] = total_grad_norm / n_batches

        concatenated_outputs = self._concatenate_outputs(all_outputs)
        concatenated_labels = self._concatenate_labels(all_labels)
        if concatenated_outputs is not None and concatenated_labels is not None:
            metrics.update(
                self.compute_metrics(concatenated_outputs, concatenated_labels)
            )

        metrics_tracker.update_train(metrics)
        return metrics

    def evaluate(self, data_loader: Any) -> tuple[Any, Any, dict[str, float]]:
        self.model.eval()
        loss_components: dict[str, float] = {}
        all_outputs: list[Any] = []
        all_labels: list[Any] = []

        if not len(data_loader):
            self.logger.warning(
                "Evaluate() called with an empty data loader. Returning None."
            )
            return None, None, {}

        n_batches = len(data_loader)
        with torch.no_grad():
            for batch in data_loader:
                features, labels = self._prepare_batch(batch)
                device_type = "cuda" if "cuda" in self.device else "cpu"
                with torch.autocast(device_type=device_type, dtype=self.dtype):
                    outputs = (
                        self.model(*features)
                        if isinstance(features, tuple)
                        else self.model(features)
                    )
                    loss_output = self.compute_loss(outputs, labels, is_training=False)

                if isinstance(loss_output, dict):
                    if not loss_components:
                        loss_components = {key: 0.0 for key in loss_output.keys()}
                    for key in loss_output.keys():
                        if key not in loss_components:
                            loss_components[key] = 0.0
                    for name, component_loss in loss_output.items():
                        if torch.is_tensor(component_loss):
                            loss_components[name] += (
                                component_loss.detach().cpu().item()
                            )
                        else:
                            loss_components[name] += component_loss
                else:
                    if "loss" not in loss_components:
                        loss_components["loss"] = 0.0
                    loss_components["loss"] += loss_output.item()

                all_outputs.append(outputs)
                all_labels.append(labels)

        concatenated_outputs = self._concatenate_outputs(all_outputs)
        concatenated_labels = self._concatenate_labels(all_labels)

        if loss_components:
            if "total" in loss_components:
                avg_losses = {
                    f"loss_{name}": total / n_batches
                    for name, total in loss_components.items()
                    if name != "total"
                }
                avg_losses["loss"] = loss_components["total"] / n_batches
            else:
                avg_losses = {
                    f"loss_{name}": total / n_batches
                    for name, total in loss_components.items()
                    if name != "loss"
                }
                avg_losses["loss"] = loss_components["loss"] / n_batches
        else:
            avg_losses = {"loss": 0.0}

        return concatenated_outputs, concatenated_labels, avg_losses

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        num_epochs: int,
        monitor_metric: str = "loss",
        mode: str = "min",
        early_stopping_patience: int = 10,
        save_dir: str | Path | None = None,
        plot_curves: bool = False,
        stage: str | None = None,
    ) -> dict[str, Any]:
        self.current_stage = stage
        self.optimizer = create_optimizer(self.model, self.config, self.logger)

        scheduler_params = getattr(self.config, "lr_scheduler", {})
        num_warmup_epochs = getattr(self.config, "warmup_epochs", 0)
        scheduler = WarmupPlateauScheduler(
            optimizer=self.optimizer,
            num_warmup_epochs=num_warmup_epochs,
            total_epochs=num_epochs,
            mode=mode,
            **scheduler_params,
        )
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            logger=self.logger,
            monitor=monitor_metric,
            mode=mode,
        )

        metrics_tracker = MetricsTracker()
        best_val_metrics: dict[str, Any] | None = None
        best_val_outputs = None
        val_labels = None
        best_checkpoint_path: Path | None = None

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            try:
                self.current_epoch = epoch
                if hasattr(self, "on_epoch_start"):
                    self.on_epoch_start(epoch, train_loader)

                train_metrics = self.train_epoch(train_loader, metrics_tracker)
                val_outputs, current_val_labels, val_losses = self.evaluate(val_loader)

                if val_outputs is None or current_val_labels is None:
                    self.logger.warning(
                        "Skipping validation metrics for epoch %s due to "
                        "empty val loader.",
                        epoch,
                    )
                    continue

                val_metrics = self.compute_metrics(val_outputs, current_val_labels)
                val_metrics.update(val_losses)
                metrics_tracker.update_val(val_metrics)

                scheduler.step(val_metrics.get(monitor_metric))
                self._log_epoch_metrics(train_metrics, val_metrics)

                is_best = early_stopping(val_metrics)
                if is_best:
                    if save_dir and (
                        self.save_checkpoints
                        or hasattr(self, "_always_save_checkpoints")
                    ):
                        best_checkpoint_path = Path(save_dir) / "best_model.pt"
                        extra_state = getattr(
                            self,
                            "_get_extra_checkpoint_state",
                            lambda: {},
                        )()
                        self.save_checkpoint(
                            Path(save_dir) / "checkpoint.pt",
                            is_best=True,
                            extra_state=extra_state,
                        )

                    best_val_metrics = val_metrics.copy()
                    best_val_outputs = val_outputs
                    if val_labels is None:
                        val_labels = current_val_labels

                if hasattr(self, "on_epoch_end"):
                    self.on_epoch_end(epoch, val_metrics, is_best=is_best)

                if early_stopping.early_stop:
                    self.logger.info("Early stopping triggered")
                    break

            except Exception as exc:
                self.logger.error("Error during epoch %s: %s", epoch, exc)
                raise

        if plot_curves and save_dir:
            try:
                self._plot_training_curves(metrics_tracker, Path(save_dir), stage)
            except Exception as exc:
                self.logger.warning("Error plotting training curves: %s", exc)

        results: dict[str, Any] = {
            "epochs_trained": epoch + 1,
            "train_metrics": train_metrics,
            "final_val_metrics": best_val_metrics,
            "val_outputs": best_val_outputs,
            "val_labels": val_labels,
        }
        if best_checkpoint_path is not None:
            results["best_checkpoint_path"] = best_checkpoint_path
        return results

    def _plot_training_curves(
        self,
        metrics_tracker: MetricsTracker,
        save_dir: Path,
        stage: str | None = None,
    ) -> None:
        try:
            from effector_bincls.plotting import plot_training_curves

            plot_training_curves(
                metrics_tracker,
                fold=self.current_fold,
                save_dir=save_dir,
                stage=stage or self.current_stage,
            )
        except ImportError:
            self.logger.warning(
                "Could not import plotting module. Skipping plot generation."
            )
        except Exception as exc:
            self.logger.warning("Unexpected error in plotting: %s", exc)

    def plot_threshold_analysis(
        self,
        outputs: Any,
        labels: Any,
        save_dir: str | Path,
        fold_number: int | None = None,
        threshold_methods: list[str] | None = None,
        target_recalls: list[float] | None = None,
    ) -> None:
        try:
            from effector_bincls.plotting import plot_threshold_analysis

            plot_threshold_analysis(
                outputs=outputs,
                labels=labels,
                save_dir=save_dir,
                fold_number=fold_number,
                threshold_methods=threshold_methods,
                target_recalls=target_recalls,
                logger=self.logger,
            )
        except ImportError:
            self.logger.warning(
                "Could not import plotting module. Skipping threshold analysis."
            )
        except Exception as exc:
            self.logger.warning("Error in threshold analysis plotting: %s", exc)


__all__ = [
    "BaseTrainer",
    "create_optimizer",
    "EarlyStopping",
    "MetricsTracker",
    "WarmupPlateauScheduler",
]
