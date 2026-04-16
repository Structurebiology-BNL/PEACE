"""Checkpoint and compatibility helpers for package-native workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from ml_collections import ConfigDict

from effector_bincls.models import SimplePredictor


def get_checkpoint_path(run_dir: Path, fold: int, is_single_stage: bool) -> Path:
    """Resolve the historical checkpoint layout for a fold."""
    if is_single_stage:
        return run_dir / f"fold_{fold}" / "checkpoint.pt"
    return run_dir / f"fold_{fold}" / "finetuning" / "checkpoint.pt"


def require_baseline_model_type(config: ConfigDict) -> None:
    model_type = getattr(getattr(config, "model", None), "type", None)
    if model_type != "simple_predictor":
        raise ValueError(
            "Baseline workflows require model.type='simple_predictor', "
            f"got '{model_type}'."
        )


def require_prototype_model_type(config: ConfigDict) -> None:
    model_type = getattr(getattr(config, "model", None), "type", None)
    if model_type != "simple":
        raise ValueError(
            f"Prototype workflows require model.type='simple', got '{model_type}'."
        )


def load_baseline_model(
    model_path: Path,
    config: ConfigDict,
    device: torch.device,
) -> torch.nn.Module:
    """Load a baseline checkpoint."""
    require_baseline_model_type(config)
    model = SimplePredictor(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        dropout_rate=config.model.dropout_rate,
        use_contrastive=getattr(config.model, "use_contrastive", False),
        encoder_hidden_dim=getattr(config.model, "encoder_hidden_dim", None),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _section_get(
    section: object | None, key: str, default: object | None = None
) -> object | None:
    if section is None:
        return default
    if hasattr(section, key):
        return getattr(section, key)
    if isinstance(section, dict):
        return section.get(key, default)
    if hasattr(section, "get"):
        return section.get(key, default)
    return default


def resolve_prototype_scoring_temperature(
    config: ConfigDict,
    *,
    is_single_stage: bool,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Resolve inference-time scoring temperature across historical config layouts."""
    if is_single_stage:
        params = _section_get(
            getattr(config, "training", None), "prototype_loss_params"
        )
        value = _section_get(params, "scoring_temperature", 1.0)
        if logger is not None:
            logger.info(
                "Using single-stage prototype scoring_temperature: %s",
                value,
            )
        return float(value)

    finetuning = _section_get(getattr(config, "training", None), "finetuning")
    hybrid_params = _section_get(finetuning, "hybrid_loss_params")
    if (
        hybrid_params is not None
        and _section_get(hybrid_params, "scoring_temperature") is not None
    ):
        value = _section_get(hybrid_params, "scoring_temperature", 1.0)
        if logger is not None:
            logger.info("Using finetuning hybrid scoring_temperature: %s", value)
        return float(value)

    prototype_params = _section_get(finetuning, "prototype_loss_params")
    value = _section_get(prototype_params, "scoring_temperature", 1.0)
    if logger is not None:
        logger.info("Using finetuning prototype scoring_temperature: %s", value)
    return float(value)


def extract_checkpoint_prototypes(
    checkpoint: dict,
    device: torch.device,
) -> torch.Tensor | None:
    """Read prototypes from top-level or extra_state historical checkpoints."""
    if checkpoint.get("prototypes") is not None:
        return checkpoint["prototypes"].to(device)
    extra_state = checkpoint.get("extra_state", {})
    if extra_state.get("prototypes") is not None:
        return extra_state["prototypes"].to(device)
    return None


def load_prototype_ranking_model(
    model_path: Path,
    config: ConfigDict,
    device: torch.device,
    is_single_stage: bool,
    logger: Optional[logging.Logger] = None,
) -> tuple[torch.nn.Module, torch.Tensor | None, float]:
    """Load a prototype checkpoint compatible with historical runs."""
    require_prototype_model_type(config)
    model = SimplePredictor(
        input_dim=getattr(config.model, "input_dim", 1024),
        output_dim=getattr(config.model, "output_dim", 1),
        dropout_rate=getattr(config.model, "dropout_rate", 0.2),
        use_contrastive=True,
        contrastive_dim=getattr(config.model, "contrastive_dim", 512),
        encoder_hidden_dim=getattr(config.model, "encoder_hidden_dim", 512),
    ).to(device)
    model.set_training_mode("pretraining")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prototypes = extract_checkpoint_prototypes(checkpoint, device)
    scoring_temperature = resolve_prototype_scoring_temperature(
        config,
        is_single_stage=is_single_stage,
        logger=logger,
    )
    return model, prototypes, scoring_temperature
