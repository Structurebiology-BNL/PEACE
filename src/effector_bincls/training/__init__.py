"""Training entrypoints and validation helpers for effector binary classification."""

from effector_bincls.training.cross_validation import (
    run_baseline_cv,
    run_prototype_ranking_cv,
    run_prototype_ranking_two_stage_cv,
)
from effector_bincls.training.cv_utils import (
    compute_aggregated_metrics,
    compute_global_threshold_optimization,
    prepare_cv_results,
    save_oof_predictions,
)
from effector_bincls.training.data import (
    create_baseline_data_loader_fn,
    create_single_stage_data_loader_fn,
    create_two_stage_data_loader_fn,
    load_test_data,
)
from effector_bincls.training.losses import (
    ConSupPrototypeLoss,
    HybridContrastiveLoss,
    PrototypeBCELoss,
)
from effector_bincls.training.runtime import (
    BaseTrainer,
    EarlyStopping,
    MetricsTracker,
    WarmupPlateauScheduler,
    create_optimizer,
)
from effector_bincls.training.trainers import (
    BaselineTrainer,
    PretrainTrainer,
    PrototypeRankingTrainer,
)
from effector_bincls.training.validation import (
    validate_baseline_training_config,
    validate_prototype_single_stage_config,
    validate_prototype_two_stage_config,
)

__all__ = [
    "validate_baseline_training_config",
    "validate_prototype_single_stage_config",
    "validate_prototype_two_stage_config",
    "run_baseline_cv",
    "run_prototype_ranking_cv",
    "run_prototype_ranking_two_stage_cv",
    "create_baseline_data_loader_fn",
    "create_single_stage_data_loader_fn",
    "create_two_stage_data_loader_fn",
    "load_test_data",
    "BaseTrainer",
    "EarlyStopping",
    "MetricsTracker",
    "WarmupPlateauScheduler",
    "create_optimizer",
    "BaselineTrainer",
    "PretrainTrainer",
    "PrototypeRankingTrainer",
    "ConSupPrototypeLoss",
    "HybridContrastiveLoss",
    "PrototypeBCELoss",
    "compute_aggregated_metrics",
    "compute_global_threshold_optimization",
    "prepare_cv_results",
    "save_oof_predictions",
]
