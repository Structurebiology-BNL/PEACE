from __future__ import annotations

import torch
from ml_collections import ConfigDict

from effector_bincls.training.runtime import BaseTrainer


class MinimalTrainer(BaseTrainer):
    def compute_loss(self, outputs, labels, is_training=True):
        return {"total": outputs.sum()}

    def compute_metrics(self, outputs, labels):
        return {}


def test_detach_output_recurses_through_tuple_and_list() -> None:
    tensor = torch.ones(2, requires_grad=True)

    detached = MinimalTrainer._detach_output((tensor, [tensor]))

    assert isinstance(detached, tuple)
    assert isinstance(detached[1], list)
    assert not detached[0].requires_grad
    assert not detached[1][0].requires_grad


def test_detach_output_preserves_non_tensor_values() -> None:
    trainer = MinimalTrainer(
        model=torch.nn.Linear(1, 1),
        config=ConfigDict(),
        device="cpu",
    )

    assert trainer._detach_output("metadata") == "metadata"
