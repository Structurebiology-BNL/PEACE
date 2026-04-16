"""Package-owned embedding dataset implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from effector_bincls.data.contracts import load_labeled_dataset, resolve_label_columns
from effector_bincls.data.packed_embeddings import (
    open_packed_embedding_dataset,
    require_sequence_indices,
)

LOGGER = logging.getLogger(__name__)


def _format_missing(prefix: str, missing_ids: list[str], source: Path) -> str:
    preview = missing_ids[:5]
    suffix = "" if len(missing_ids) <= 5 else f" and {len(missing_ids) - 5} more"
    return f"{prefix} in {source}: {preview}{suffix}"


class BaseProteinDataset(Dataset, ABC):
    """Base dataset for pooled protein embeddings."""

    def __init__(
        self,
        embedding_dir: str | Path,
        sequence_ids: list[str],
        normalize: bool = True,
        pooling_type: str = "mean",
        use_variants: bool = False,
        csv_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.embedding_dir = Path(embedding_dir)
        self.sequence_ids = [str(sequence_id) for sequence_id in sequence_ids]
        self.normalize = normalize
        self.pooling_type = pooling_type
        self.use_variants = use_variants
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.variant_metadata: dict[str, object] | None = None

        if pooling_type not in {"mean", "max", "bos", "eos"}:
            raise ValueError("pooling_type must be one of: 'mean', 'max', 'bos', 'eos'")

        if not self.sequence_ids:
            raise ValueError("At least one sequence ID is required")

        self.packed_embeddings, packed_sequence_ids, self.variant_metadata = (
            open_packed_embedding_dataset(self.embedding_dir)
        )
        dataset_pooling_type = self.variant_metadata.get("pooling_type")
        if dataset_pooling_type != self.pooling_type:
            raise ValueError(
                "Packed embedding dataset pooling_type does not match request: "
                f"expected {self.pooling_type!r}, got {dataset_pooling_type!r}"
            )
        self.seq_to_packed_idx = require_sequence_indices(
            self.sequence_ids,
            packed_sequence_ids,
        )
        self.original_variant_index = int(
            self.variant_metadata.get("original_variant_index", 0)
        )
        if not 0 <= self.original_variant_index < self.packed_embeddings.shape[1]:
            raise ValueError(
                "Packed embedding dataset original_variant_index is out of bounds"
            )

    def __len__(self) -> int:
        return len(self.sequence_ids)

    @abstractmethod
    def get_labels(self, idx: int) -> torch.Tensor:
        """Return the label tensor for one dataset item."""

    def load_and_process_embeddings(
        self,
        sequence_id: str,
    ) -> torch.Tensor:
        """Load one packed embedding row and apply optional normalization."""
        packed_index = self.seq_to_packed_idx[sequence_id]
        embeddings = self.packed_embeddings[packed_index]
        if self.use_variants:
            tensor = torch.tensor(embeddings, dtype=torch.float32)
        else:
            tensor = torch.tensor(
                embeddings[self.original_variant_index],
                dtype=torch.float32,
            )

        if self.normalize:
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)

        return tensor


class SimpleDataset(BaseProteinDataset):
    """Embedding-backed labeled dataset for supported binary workflows."""

    def __init__(
        self,
        embedding_dir: str | Path,
        csv_path: str | Path,
        sequence_ids: list[str] | None = None,
        normalize: bool = True,
        pooling_type: str = "mean",
        use_variants: bool = False,
        label_config: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.labels_df = load_labeled_dataset(csv_path, label_config=label_config)
        self.sequence_id_column, self.label_column = resolve_label_columns(label_config)
        self.labels_dict = dict(
            zip(
                self.labels_df[self.sequence_id_column],
                self.labels_df[self.label_column],
                strict=False,
            )
        )

        if sequence_ids is None:
            sequence_ids = self.labels_df[self.sequence_id_column].tolist()
        else:
            sequence_ids = [str(sequence_id) for sequence_id in sequence_ids]

        missing_labels = sorted(set(sequence_ids) - set(self.labels_dict))
        if missing_labels:
            raise ValueError(
                _format_missing(
                    "Sequence IDs not found in labels CSV",
                    missing_labels,
                    Path(csv_path),
                )
            )

        super().__init__(
            embedding_dir=embedding_dir,
            sequence_ids=sequence_ids,
            normalize=normalize,
            pooling_type=pooling_type,
            use_variants=use_variants,
            csv_path=csv_path,
            logger=self.logger,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_id = self.sequence_ids[idx]
        embeddings = self.load_and_process_embeddings(sequence_id)
        return embeddings, self.get_labels(idx)

    def get_labels(self, idx: int) -> torch.Tensor:
        sequence_id = self.sequence_ids[idx]
        return torch.tensor([self.labels_dict[sequence_id]], dtype=torch.float32)
