import torch

from effector_bincls.prototype_scoring import (
    compute_prototype_distance_scores,
    compute_prototype_probabilities,
)


def test_compute_prototype_scores_for_2d_embeddings() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    prototypes = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    scores = compute_prototype_distance_scores(embeddings, prototypes, 0.5)
    probabilities = compute_prototype_probabilities(embeddings, prototypes, 0.5)

    assert torch.allclose(scores, torch.tensor([4.0, 0.0]))
    assert torch.allclose(
        probabilities,
        torch.tensor([0.98201376, 0.5]),
        atol=1e-6,
    )


def test_compute_prototype_scores_for_variant_embeddings() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    prototypes = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    scores = compute_prototype_distance_scores(embeddings, prototypes, 1.2)
    probabilities = compute_prototype_probabilities(embeddings, prototypes, 1.2)

    assert torch.allclose(scores, torch.tensor([1.6418, 0.2021]), atol=1e-4)
    assert torch.allclose(
        probabilities,
        torch.tensor([0.8378, 0.5504]),
        atol=1e-4,
    )
