"""Package-owned packed embedding dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike

FORMAT_VERSION = 1
LAYOUT = "variant"


def _find_duplicate_ids(sequence_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sequence_id in sequence_ids:
        if sequence_id in seen and sequence_id not in duplicates:
            duplicates.append(sequence_id)
        seen.add(sequence_id)
    return duplicates


def _validate_sequence_ids(sequence_ids: list[str]) -> None:
    duplicates = _find_duplicate_ids(sequence_ids)
    if duplicates:
        raise ValueError(f"Found duplicate sequence IDs: {duplicates}")


def _read_sequence_ids(path: Path) -> list[str]:
    contents = path.read_text()
    if not contents:
        return []
    return contents.splitlines()


def _validate_metadata(
    metadata: dict[str, object],
    embeddings: np.memmap,
    sequence_ids: list[str],
) -> None:
    duplicates = _find_duplicate_ids(sequence_ids)
    if duplicates:
        raise ValueError(f"Found duplicate sequence IDs: {duplicates}")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Packed embedding dataset has unsupported format_version: "
            f"{metadata.get('format_version')}"
        )
    if metadata.get("layout") != LAYOUT:
        raise ValueError(
            "Packed embedding dataset must use layout "
            f"'{LAYOUT}', got {metadata.get('layout')!r}"
        )
    if embeddings.ndim != 3:
        raise ValueError(
            "Packed embeddings must have shape [N, V, D]; "
            f"received {embeddings.shape}"
        )
    if len(sequence_ids) != embeddings.shape[0]:
        raise ValueError(
            "Packed embedding dataset sequence IDs do not match array rows"
        )
    expected_counts = {
        "num_sequences": embeddings.shape[0],
        "num_variants": embeddings.shape[1],
        "embedding_dim": embeddings.shape[2],
    }
    for field_name, expected_value in expected_counts.items():
        if metadata.get(field_name) != expected_value:
            raise ValueError(
                "Packed embedding dataset metadata "
                f"{field_name} does not match array shape"
            )


def require_sequence_indices(
    requested_ids: list[str],
    available_ids: list[str],
) -> dict[str, int]:
    """Resolve requested IDs to packed dataset indices."""
    available = [str(sequence_id) for sequence_id in available_ids]
    requested = [str(sequence_id) for sequence_id in requested_ids]

    _validate_sequence_ids(available)

    index_by_id = {sequence_id: index for index, sequence_id in enumerate(available)}
    missing_ids = [
        sequence_id for sequence_id in requested if sequence_id not in index_by_id
    ]
    if missing_ids:
        raise FileNotFoundError(f"Missing embeddings for sequence IDs: {missing_ids}")
    return {sequence_id: index_by_id[sequence_id] for sequence_id in requested}


def _write_sequence_ids(dataset_path: Path, sequence_ids: list[str]) -> None:
    sequence_ids_text = "\n".join(sequence_ids)
    if sequence_ids_text:
        sequence_ids_text += "\n"
    (dataset_path / "sequence_ids.txt").write_text(sequence_ids_text)


def _write_metadata(
    dataset_path: Path,
    shape: tuple[int, int, int],
    *,
    pooling_type: str,
    original_variant_index: int,
    dtype: np.dtype,
) -> None:
    metadata = {
        "format_version": FORMAT_VERSION,
        "layout": LAYOUT,
        "num_sequences": shape[0],
        "num_variants": shape[1],
        "embedding_dim": shape[2],
        "pooling_type": pooling_type,
        "original_variant_index": original_variant_index,
        "dtype": str(dtype),
    }
    (dataset_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def create_packed_embedding_memmap(
    dataset_dir: str | Path,
    sequence_ids: list[str],
    *,
    shape: tuple[int, int, int],
    dtype: DTypeLike,
    pooling_type: str,
    original_variant_index: int = 0,
) -> np.memmap:
    """Create a packed embedding dataset and return a row-writable memmap."""
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    normalized_sequence_ids = [str(sequence_id) for sequence_id in sequence_ids]
    _validate_sequence_ids(normalized_sequence_ids)

    if len(shape) != 3:
        raise ValueError(f"Packed embeddings must use [N, V, D] shape; got {shape}")
    if shape[0] != len(normalized_sequence_ids):
        raise ValueError(
            "Packed embeddings first dimension must match number of sequence IDs"
        )
    if not 0 <= original_variant_index < shape[1]:
        raise ValueError("original_variant_index must be within the variant range")

    embedding_dtype = np.dtype(dtype)
    embeddings = np.lib.format.open_memmap(
        dataset_path / "embeddings.npy",
        mode="w+",
        dtype=embedding_dtype,
        shape=shape,
    )
    _write_sequence_ids(dataset_path, normalized_sequence_ids)
    _write_metadata(
        dataset_path,
        shape,
        pooling_type=pooling_type,
        original_variant_index=original_variant_index,
        dtype=embedding_dtype,
    )
    return embeddings


def write_packed_embedding_dataset(
    dataset_dir: str | Path,
    sequence_ids: list[str],
    embeddings: np.ndarray,
    *,
    pooling_type: str,
    original_variant_index: int = 0,
) -> Path:
    """Write a packed variant-layout embedding dataset."""
    embedding_array = np.asarray(embeddings)
    if embedding_array.ndim != 3:
        raise ValueError(
            "Packed embeddings must have shape [N, V, D]; "
            f"received {embedding_array.shape}"
        )
    writer = create_packed_embedding_memmap(
        dataset_dir,
        sequence_ids,
        shape=embedding_array.shape,
        dtype=embedding_array.dtype,
        pooling_type=pooling_type,
        original_variant_index=original_variant_index,
    )
    writer[:] = embedding_array
    writer.flush()
    return Path(dataset_dir)


def open_packed_embedding_dataset(
    dataset_dir: str | Path,
    mmap_mode: str = "r",
) -> tuple[np.memmap, list[str], dict[str, object]]:
    """Open a packed variant-layout embedding dataset."""
    dataset_path = Path(dataset_dir)
    embeddings = np.load(dataset_path / "embeddings.npy", mmap_mode=mmap_mode)
    sequence_ids = _read_sequence_ids(dataset_path / "sequence_ids.txt")
    with (dataset_path / "metadata.json").open() as metadata_file:
        metadata = json.load(metadata_file)
    _validate_metadata(metadata, embeddings, sequence_ids)
    return embeddings, sequence_ids, metadata
