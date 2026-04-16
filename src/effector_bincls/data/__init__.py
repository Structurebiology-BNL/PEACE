"""Package-owned data contracts and dataset helpers."""

from effector_bincls.data.contracts import (
    DEFAULT_PARTITION_COLUMN,
    load_labeled_dataset,
    resolve_label_columns,
    validate_two_stage_dataset_pair,
)
from effector_bincls.data.datasets import SimpleDataset
from effector_bincls.data.packed_embeddings import (
    create_packed_embedding_memmap,
    open_packed_embedding_dataset,
    require_sequence_indices,
    write_packed_embedding_dataset,
)

__all__ = [
    "DEFAULT_PARTITION_COLUMN",
    "SimpleDataset",
    "create_packed_embedding_memmap",
    "load_labeled_dataset",
    "open_packed_embedding_dataset",
    "require_sequence_indices",
    "resolve_label_columns",
    "validate_two_stage_dataset_pair",
    "write_packed_embedding_dataset",
]
