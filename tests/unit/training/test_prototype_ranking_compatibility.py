import pytest
from ml_collections import ConfigDict

from effector_bincls.models import SimplePredictor
from effector_bincls.training.trainers import PrototypeRankingTrainer


def make_model() -> SimplePredictor:
    return SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=4,
        encoder_hidden_dim=8,
    )


def make_finetuning_config(loss_overrides: dict | None = None) -> ConfigDict:
    prototype_loss_params = ConfigDict(
        {
            "temperature": 0.07,
            "eps_pos": 0.2,
            "eps_neg": 0.1,
            "prototype_weight": 1.0,
            "unsupervised_weight": 0.4,
            "prototype_update_strategy": "warmup_freeze",
            "prototype_momentum": 0.97,
            "scoring_temperature": 0.35,
            "ranking_weight": 0.8,
            "ranking_loss_type": "bce",
        }
    )
    if loss_overrides:
        for key, value in loss_overrides.items():
            prototype_loss_params[key] = value

    return ConfigDict(
        {
            "contrastive_type": "prototype_ranking",
            "monitor_metric": "auprc",
            "mode": "max",
            "prototype_loss_params": prototype_loss_params,
        }
    )


def test_trainer_uses_historical_scoring_temperature_and_weight() -> None:
    trainer = PrototypeRankingTrainer(
        model=make_model(),
        config=make_finetuning_config(),
        device="cpu",
    )

    assert trainer.contrastive_loss_fn.scoring_temperature == pytest.approx(0.35)
    assert trainer.contrastive_loss_fn.bce_weight == pytest.approx(0.8)


def test_prototype_ranking_trainer_prefers_explicit_bce_weight() -> None:
    trainer = PrototypeRankingTrainer(
        model=make_model(),
        config=make_finetuning_config(
            {
                "ranking_weight": 0.8,
                "bce_weight": 0.6,
            }
        ),
        device="cpu",
    )

    assert trainer.contrastive_loss_fn.bce_weight == pytest.approx(0.6)


def test_prototype_ranking_trainer_rejects_non_bce_ranking_loss_type() -> None:
    with pytest.raises(ValueError, match="ranking_loss_type='bce'"):
        PrototypeRankingTrainer(
            model=make_model(),
            config=make_finetuning_config({"ranking_loss_type": "pairwise"}),
            device="cpu",
        )
