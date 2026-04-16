import pytest
import torch

from effector_bincls.models import SimplePredictor


def test_simple_predictor_rejects_invalid_training_mode() -> None:
    model = SimplePredictor()

    with pytest.raises(
        ValueError, match="Training mode must be 'pretraining' or 'finetuning'"
    ):
        model.set_training_mode("invalid")


def test_simple_predictor_baseline_state_dict_contract() -> None:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=False,
        encoder_hidden_dim=8,
    )

    state_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }

    assert state_shapes == {
        "shared_encoder.0.weight": (8,),
        "shared_encoder.0.bias": (8,),
        "shared_encoder.2.weight": (8, 8),
        "shared_encoder.2.bias": (8,),
        "classification_head.0.weight": (8,),
        "classification_head.0.bias": (8,),
        "classification_head.1.weight": (1, 8),
        "classification_head.1.bias": (1,),
    }


def test_simple_predictor_prototype_state_dict_contract() -> None:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=4,
        encoder_hidden_dim=8,
    )

    state_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }

    assert state_shapes == {
        "shared_encoder.0.weight": (8,),
        "shared_encoder.0.bias": (8,),
        "shared_encoder.2.weight": (8, 8),
        "shared_encoder.2.bias": (8,),
        "classification_head.0.weight": (8,),
        "classification_head.0.bias": (8,),
        "classification_head.1.weight": (1, 8),
        "classification_head.1.bias": (1,),
        "contrastive_head.0.weight": (8,),
        "contrastive_head.0.bias": (8,),
        "contrastive_head.1.weight": (4, 8),
        "contrastive_head.1.bias": (4,),
    }


def test_simple_predictor_baseline_forward_contract() -> None:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=False,
        encoder_hidden_dim=8,
    )
    inputs = torch.randn(3, 8)

    logits, features = model(inputs, return_features=True)

    assert logits.shape == (3, 1)
    assert features.shape == (3, 8)


def test_simple_predictor_prototype_forward_contract_with_variants() -> None:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=4,
        encoder_hidden_dim=8,
    )
    model.set_training_mode("finetuning")
    inputs = torch.randn(2, 3, 8)

    logits, embeddings, features = model(inputs, return_features=True)

    assert logits.shape == (2, 1)
    assert embeddings.shape == (2, 3, 4)
    assert features.shape == (2, 3, 8)


def test_simple_predictor_freeze_helpers_toggle_requires_grad() -> None:
    model = SimplePredictor(
        input_dim=8,
        output_dim=1,
        dropout_rate=0.1,
        use_contrastive=True,
        contrastive_dim=4,
        encoder_hidden_dim=8,
    )

    model.freeze_encoder(True)
    model.freeze_classification_head(True)
    model.freeze_contrastive_head(True)

    assert all(not param.requires_grad for param in model.shared_encoder.parameters())
    assert all(
        not param.requires_grad for param in model.classification_head.parameters()
    )
    assert all(not param.requires_grad for param in model.contrastive_head.parameters())

    model.freeze_encoder(False)
    model.freeze_classification_head(False)
    model.freeze_contrastive_head(False)

    assert all(param.requires_grad for param in model.shared_encoder.parameters())
    assert all(param.requires_grad for param in model.classification_head.parameters())
    assert all(param.requires_grad for param in model.contrastive_head.parameters())
