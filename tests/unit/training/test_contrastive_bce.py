from __future__ import annotations

import pytest
import torch
from ml_collections import ConfigDict

from effector_bincls.models import SimplePredictor
from effector_bincls.training.contrastive_bce_trainer import (
    ContrastiveBCETrainer,
)


def make_model() -> SimplePredictor:
    return SimplePredictor(
        input_dim=4,
        output_dim=1,
        dropout_rate=0.0,
        use_contrastive=True,
        contrastive_dim=2,
        encoder_hidden_dim=4,
    )


def make_config(
    *,
    bce_weight: float = 1.0,
    unsupervised_weight: float = 0.5,
) -> ConfigDict:
    return ConfigDict(
        {
            "bce_weight": bce_weight,
            "unsupervised_weight": unsupervised_weight,
            "temperature": 0.07,
            "monitor_metric": "auprc",
            "mode": "max",
        }
    )


def make_trainer(
    *,
    bce_weight: float = 1.0,
    unsupervised_weight: float = 0.5,
) -> ContrastiveBCETrainer:
    return ContrastiveBCETrainer(
        model=make_model(),
        config=make_config(
            bce_weight=bce_weight,
            unsupervised_weight=unsupervised_weight,
        ),
        device="cpu",
    )


def test_contrastive_bce_combines_bce_and_infonce() -> None:
    trainer = make_trainer(bce_weight=2.0, unsupervised_weight=0.5)
    logits = torch.tensor([[0.2], [-0.1]], requires_grad=True)
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1]],
            [[0.0, 1.0], [0.1, 0.9]],
        ],
        requires_grad=True,
    )

    losses = trainer.compute_loss(
        (logits, embeddings),
        torch.tensor([1, 0]),
    )

    assert losses["contrastive"].item() > 0.0
    assert torch.allclose(
        losses["total"].detach(),
        2.0 * losses["bce"] + 0.5 * losses["contrastive"],
    )
    losses["total"].backward()
    assert logits.grad is not None
    assert embeddings.grad is not None


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        (torch.randn(2, 4), "requires 3D contrastive embeddings"),
        (torch.randn(2, 1, 4), "requires at least two views"),
    ],
)
def test_contrastive_bce_rejects_invalid_view_shapes(
    embeddings: torch.Tensor,
    message: str,
) -> None:
    trainer = make_trainer()

    with pytest.raises(ValueError, match=message):
        trainer.compute_loss(
            (torch.randn(2, 1), embeddings),
            torch.tensor([1, 0]),
        )


def test_contrastive_bce_requires_tuple_model_output() -> None:
    trainer = make_trainer()

    with pytest.raises(ValueError, match="expects model output"):
        trainer.compute_loss(torch.randn(2, 1), torch.tensor([1, 0]))


def test_contrastive_bce_metrics_use_classification_logits() -> None:
    trainer = make_trainer()
    logits = torch.tensor([[3.0], [-3.0]])
    embeddings = torch.randn(2, 2, 4)

    metrics = trainer.compute_metrics(
        (logits, embeddings),
        torch.tensor([1, 0]),
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)


def test_contrastive_bce_train_fold_uses_classification_logits(tmp_path) -> None:
    trainer = make_trainer()
    logits = torch.tensor([[3.0], [-3.0]])
    embeddings = torch.randn(2, 2, 4)
    labels = torch.tensor([1, 0])

    def fake_train(**_kwargs):
        return {
            "epochs_trained": 1,
            "val_outputs": (logits, embeddings),
            "val_labels": labels,
        }

    trainer.train = fake_train
    trainer.plot_threshold_analysis = lambda **_kwargs: None

    fold_results = trainer.train_fold(
        fold_number=1,
        train_loader=None,
        val_loader=None,
        save_dir=tmp_path,
        plot_curves=False,
    )

    assert fold_results["fold"] == 1
    assert fold_results["val_predictions"].shape == (2,)
    assert fold_results["val_metrics"]["roc_auc"] == pytest.approx(1.0)
