"""Shared prototype checkpoint loading for package-native entrypoints."""

from effector_bincls.checkpoints import (
    load_prototype_ranking_model,
    resolve_prototype_scoring_temperature,
)

__all__ = [
    "load_prototype_ranking_model",
    "resolve_prototype_scoring_temperature",
]
