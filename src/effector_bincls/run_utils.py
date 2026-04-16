"""Shared runtime helpers for package-native workflows."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must deserialize to a mapping: {config_path}")

    return config


def load_run_config(run_dir: str | Path) -> ConfigDict:
    """Load the saved package run configuration."""
    return ConfigDict(load_config(Path(run_dir) / "config.yml"))


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/torch values into YAML-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [convert_to_serializable(value) for value in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    return obj


def convert_config_to_dict(config: Any) -> Any:
    """Recursively convert ConfigDict-style objects into builtin containers."""
    if isinstance(config, dict):
        return {key: convert_config_to_dict(value) for key, value in config.items()}
    if isinstance(config, list):
        return [convert_config_to_dict(value) for value in config]
    if isinstance(config, tuple):
        return [convert_config_to_dict(value) for value in config]
    if hasattr(config, "items"):
        return {key: convert_config_to_dict(value) for key, value in config.items()}
    if hasattr(config, "__dict__"):
        return {
            key: convert_config_to_dict(value)
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return config


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def resolve_device(config: ConfigDict) -> torch.device:
    """Resolve the torch device from the saved hardware config."""
    gpu_id = getattr(config.hardware, "gpu_id", -1)
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def setup_logger(
    output_dir: str | Path,
    name: str | None = None,
    *,
    use_rotating_file: bool = True,
    log_file_name: str = "train",
    log_level: str = "INFO",
) -> logging.Logger:
    """Create a logger with console and file handlers."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{log_file_name}.log"

    if use_rotating_file:
        file_handler: logging.Handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
    else:
        file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_config_params(config: ConfigDict, logger: logging.Logger) -> None:
    """Log a flattened configuration summary."""
    flattened: dict[str, Any] = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flattened[f"{section}.{key}"] = value
        else:
            flattened[section] = params

    for key in sorted(flattened):
        logger.info("config.%s = %s", key, flattened[key])


def setup_training(
    config_path: str | Path,
    *,
    run_suffix: str = "",
    use_rotating_logger: bool = True,
) -> tuple[ConfigDict, Path, logging.Logger]:
    """Load a training config, create a run directory, and save the effective config."""
    config_dict = load_config(config_path)
    config = ConfigDict(config_dict)

    root_dir = Path(config.data.results_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    model_dir = root_dir / config.model.type.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_seed{config.hardware.random_seed}"
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    run_dir = model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_level = "DEBUG" if getattr(config.hardware, "debug_logging", False) else "INFO"
    logger = setup_logger(
        output_dir=run_dir,
        use_rotating_file=use_rotating_logger,
        log_level=log_level,
    )

    config_save_path = run_dir / "config.yml"
    with config_save_path.open("w") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)
    logger.info("Configuration saved to %s", config_save_path)
    logger.info("\nConfiguration:\n%s", yaml.safe_dump(config_dict, sort_keys=False))

    seed_everything(
        config.hardware.random_seed,
        deterministic=getattr(config.hardware, "deterministic", False),
    )
    logger.info("Using device: %s", resolve_device(config))
    return config, run_dir, logger


def cleanup_checkpoints(
    fold_dir: str | Path,
    logger: logging.Logger,
    fold_number: int,
    checkpoint_paths: list[str | Path] | None = None,
) -> None:
    """Remove fold checkpoints when save_checkpoints is disabled."""
    fold_dir = Path(fold_dir)
    deleted_files: list[Path] = []

    if checkpoint_paths:
        candidate_dirs = [Path(path).parent for path in checkpoint_paths if path]
    else:
        candidate_dirs = [
            fold_dir / "prototype_ranking",
            fold_dir / "pretraining",
            fold_dir / "finetuning",
            fold_dir / "training",
        ]

    for checkpoint_dir in candidate_dirs:
        if not checkpoint_dir.exists():
            continue
        for filename in ("checkpoint.pt", "best_model.pt"):
            path = checkpoint_dir / filename
            if path.exists() or path.is_symlink():
                path.unlink()
                deleted_files.append(path)

    if deleted_files:
        logger.info(
            "Fold %s: cleaned up %s checkpoint file(s)",
            fold_number,
            len(deleted_files),
        )
