"""Temporary helpers for migrating legacy per-sequence NPZ embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from effector_bincls.data.packed_embeddings import create_packed_embedding_memmap


def _select_final_layer(array: np.ndarray) -> np.ndarray:
    """Collapse legacy [L, D] arrays to the final layer while preserving [D]."""
    embedding = np.asarray(array)
    if embedding.ndim == 1:
        return embedding.astype(np.float32, copy=False)
    return embedding[-1].astype(np.float32, copy=False)


def _sorted_variant_keys(handle: np.lib.npyio.NpzFile, pooling_type: str) -> list[str]:
    variant_keys = sorted(
        (
            key
            for key in handle.files
            if key.startswith(f"{pooling_type}_variant_")
        ),
        key=lambda key: int(key.rsplit("_", maxsplit=1)[-1]),
    )
    if variant_keys:
        return variant_keys
    if pooling_type in handle.files:
        return [pooling_type]
    raise ValueError(
        f"No embeddings found for pooling type '{pooling_type}' in legacy NPZ file"
    )


def _load_packed_row(
    handle: np.lib.npyio.NpzFile,
    variant_keys: list[str],
) -> np.ndarray:
    return np.stack(
        [_select_final_layer(handle[key]) for key in variant_keys],
        axis=0,
    )


def migrate_npz_directory_to_packed_dataset(
    npz_dir: str | Path,
    output_dir: str | Path,
    *,
    pooling_type: str = "mean",
    original_variant_index: int = 0,
) -> Path:
    """Convert a legacy NPZ embedding directory into packed format."""
    npz_path = Path(npz_dir)
    output_path = Path(output_dir)
    npz_files = sorted(npz_path.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {npz_path}")

    sequence_ids = [npz_file.stem for npz_file in npz_files]
    expected_variant_keys: list[str] | None = None
    embedding_dim: int | None = None

    for npz_file in npz_files:
        with np.load(npz_file) as handle:
            variant_keys = _sorted_variant_keys(handle, pooling_type)
            if expected_variant_keys is None:
                expected_variant_keys = variant_keys
                if not 0 <= original_variant_index < len(expected_variant_keys):
                    raise ValueError(
                        "original_variant_index must be within the discovered "
                        "variant range"
                    )
            elif variant_keys != expected_variant_keys:
                raise ValueError(
                    "Legacy NPZ files do not share the same variant layout: "
                    f"{npz_file}"
                )

            row = _load_packed_row(handle, variant_keys)
            if row.ndim != 2:
                raise ValueError(
                    f"Legacy NPZ file has invalid embedding shape: {npz_file}"
                )
            if embedding_dim is None:
                embedding_dim = row.shape[1]
            elif row.shape[1] != embedding_dim:
                raise ValueError(
                    "Legacy NPZ files do not share the same embedding dimension: "
                    f"{npz_file}"
                )

    if expected_variant_keys is None or embedding_dim is None:
        raise ValueError(f"No embeddings found in {npz_path}")

    writer = create_packed_embedding_memmap(
        output_path,
        sequence_ids,
        shape=(len(npz_files), len(expected_variant_keys), embedding_dim),
        dtype=np.float32,
        pooling_type=pooling_type,
        original_variant_index=original_variant_index,
    )
    for row_index, npz_file in enumerate(npz_files):
        with np.load(npz_file) as handle:
            writer[row_index] = _load_packed_row(handle, expected_variant_keys)
    writer.flush()
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the temporary NPZ migration helper."""
    parser = argparse.ArgumentParser(
        description="Convert legacy per-sequence NPZ embeddings into packed format."
    )
    parser.add_argument(
        "--npz_dir",
        type=Path,
        required=True,
        help="Directory containing legacy per-sequence NPZ embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for the packed embedding dataset.",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="mean",
        choices=["mean", "max", "bos", "eos"],
        help="Pooling type to migrate from the legacy NPZ files.",
    )
    parser.add_argument(
        "--original_variant_index",
        type=int,
        default=0,
        help="Variant index corresponding to the original embedding view.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the temporary NPZ migration helper."""
    args = parse_args()
    migrate_npz_directory_to_packed_dataset(
        args.npz_dir,
        args.output_dir,
        pooling_type=args.pooling_type,
        original_variant_index=args.original_variant_index,
    )


if __name__ == "__main__":
    main()
