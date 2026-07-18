from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from effector_bincls.data.npz_migration import migrate_npz_directory_to_packed_dataset
from effector_bincls.data.packed_embeddings import (
    create_packed_embedding_memmap,
    open_packed_embedding_dataset,
    require_sequence_indices,
    write_packed_embedding_dataset,
)


def _write_test_packed_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "packed"
    write_packed_embedding_dataset(
        dataset_dir,
        ["seq0"],
        np.zeros((1, 2, 3), dtype=np.float32),
        pooling_type="mean",
    )
    return dataset_dir


def _replace_metadata_field(
    dataset_dir: Path,
    field_name: str,
    value: object,
) -> None:
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field_name] = value
    metadata_path.write_text(json.dumps(metadata))


def test_write_and_open_packed_embedding_dataset_round_trips(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    sequence_ids = ["seq0", "seq1"]
    embeddings = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

    written_path = write_packed_embedding_dataset(
        dataset_dir,
        sequence_ids,
        embeddings,
        pooling_type="mean",
        original_variant_index=1,
    )

    loaded_embeddings, loaded_sequence_ids, metadata = open_packed_embedding_dataset(
        written_path
    )

    assert written_path == dataset_dir
    assert isinstance(loaded_embeddings, np.memmap)
    np.testing.assert_array_equal(loaded_embeddings, embeddings)
    assert loaded_sequence_ids == sequence_ids
    assert metadata == {
        "format_version": 1,
        "layout": "variant",
        "num_sequences": 2,
        "num_variants": 2,
        "embedding_dim": 2,
        "pooling_type": "mean",
        "original_variant_index": 1,
        "dtype": "float32",
    }


def test_require_sequence_indices_rejects_missing_requested_ids() -> None:
    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        require_sequence_indices(["seq0", "seq2"], ["seq0", "seq1"])


def test_write_and_open_empty_packed_embedding_dataset_round_trips(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((0, 2, 3), dtype=np.float32)

    write_packed_embedding_dataset(
        dataset_dir,
        [],
        embeddings,
        pooling_type="mean",
    )

    loaded_embeddings, loaded_sequence_ids, metadata = open_packed_embedding_dataset(
        dataset_dir
    )

    assert loaded_embeddings.shape == (0, 2, 3)
    assert loaded_sequence_ids == []
    assert metadata["num_sequences"] == 0


def test_create_packed_embedding_memmap_supports_incremental_writes(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    writer = create_packed_embedding_memmap(
        dataset_dir,
        ["seq0", "seq1"],
        shape=(2, 2, 3),
        dtype=np.float32,
        pooling_type="mean",
        original_variant_index=1,
    )
    writer[0] = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    writer[1] = np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    writer.flush()
    del writer

    embeddings, sequence_ids, metadata = open_packed_embedding_dataset(dataset_dir)

    assert sequence_ids == ["seq0", "seq1"]
    np.testing.assert_array_equal(
        embeddings,
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert metadata["original_variant_index"] == 1


def test_open_packed_embedding_dataset_rejects_sequence_id_mismatch(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((2, 1, 3), dtype=np.float32)

    write_packed_embedding_dataset(
        dataset_dir,
        ["seq0", "seq1"],
        embeddings,
        pooling_type="mean",
    )

    (dataset_dir / "sequence_ids.txt").write_text("seq0\n")

    with pytest.raises(ValueError, match="sequence IDs"):
        open_packed_embedding_dataset(dataset_dir)


def test_open_packed_embedding_dataset_rejects_duplicate_sequence_ids(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((2, 1, 3), dtype=np.float32)

    write_packed_embedding_dataset(
        dataset_dir,
        ["seq0", "seq1"],
        embeddings,
        pooling_type="mean",
    )

    (dataset_dir / "sequence_ids.txt").write_text("seq0\nseq0\n")

    with pytest.raises(ValueError, match="duplicate sequence IDs"):
        open_packed_embedding_dataset(dataset_dir)


def test_open_packed_embedding_dataset_rejects_malformed_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((1, 2, 3), dtype=np.float32)

    write_packed_embedding_dataset(
        dataset_dir,
        ["seq0"],
        embeddings,
        pooling_type="mean",
    )

    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "layout": "packed",
                "num_sequences": 1,
                "num_variants": 99,
                "embedding_dim": 3,
                "pooling_type": "mean",
                "original_variant_index": 0,
                "dtype": "float32",
            }
        )
    )

    with pytest.raises(ValueError, match="layout|num_variants"):
        open_packed_embedding_dataset(dataset_dir)


def test_open_packed_embedding_dataset_rejects_unsupported_format_version(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((1, 1, 3), dtype=np.float32)

    write_packed_embedding_dataset(
        dataset_dir,
        ["seq0"],
        embeddings,
        pooling_type="mean",
    )

    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 999,
                "layout": "variant",
                "num_sequences": 1,
                "num_variants": 1,
                "embedding_dim": 3,
                "pooling_type": "mean",
                "original_variant_index": 0,
                "dtype": "float32",
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported format_version"):
        open_packed_embedding_dataset(dataset_dir)


@pytest.mark.parametrize(
    "field_name",
    [
        "format_version",
        "layout",
        "num_sequences",
        "num_variants",
        "embedding_dim",
        "pooling_type",
        "original_variant_index",
        "dtype",
    ],
)
def test_open_packed_embedding_dataset_requires_every_metadata_field(
    tmp_path: Path,
    field_name: str,
) -> None:
    dataset_dir = _write_test_packed_dataset(tmp_path)
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata[field_name]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=f"missing required.*{field_name}"):
        open_packed_embedding_dataset(dataset_dir)


@pytest.mark.parametrize(
    ("field_name", "value", "expected_type"),
    [
        ("format_version", True, "integer"),
        ("num_sequences", 1.0, "integer"),
        ("num_variants", True, "integer"),
        ("embedding_dim", 3.0, "integer"),
        ("original_variant_index", False, "integer"),
        ("layout", 1, "string"),
        ("pooling_type", 1, "string"),
        ("dtype", 1, "string"),
    ],
)
def test_open_packed_embedding_dataset_rejects_wrong_metadata_types(
    tmp_path: Path,
    field_name: str,
    value: object,
    expected_type: str,
) -> None:
    dataset_dir = _write_test_packed_dataset(tmp_path)
    _replace_metadata_field(dataset_dir, field_name, value)

    with pytest.raises(ValueError, match=f"{field_name}.*{expected_type}"):
        open_packed_embedding_dataset(dataset_dir)


@pytest.mark.parametrize("field_name", ["pooling_type", "dtype"])
def test_open_packed_embedding_dataset_rejects_empty_metadata_strings(
    tmp_path: Path,
    field_name: str,
) -> None:
    dataset_dir = _write_test_packed_dataset(tmp_path)
    _replace_metadata_field(dataset_dir, field_name, "")

    with pytest.raises(ValueError, match=f"{field_name}.*non-empty"):
        open_packed_embedding_dataset(dataset_dir)


@pytest.mark.parametrize("original_variant_index", [-1, 2])
def test_open_packed_embedding_dataset_rejects_invalid_original_variant_index(
    tmp_path: Path,
    original_variant_index: int,
) -> None:
    dataset_dir = _write_test_packed_dataset(tmp_path)
    _replace_metadata_field(
        dataset_dir,
        "original_variant_index",
        original_variant_index,
    )

    with pytest.raises(ValueError, match="original_variant_index.*variant range"):
        open_packed_embedding_dataset(dataset_dir)


@pytest.mark.parametrize("dtype", ["not-a-dtype", "float64"])
def test_open_packed_embedding_dataset_rejects_invalid_or_mismatched_dtype(
    tmp_path: Path,
    dtype: str,
) -> None:
    dataset_dir = _write_test_packed_dataset(tmp_path)
    _replace_metadata_field(dataset_dir, "dtype", dtype)

    with pytest.raises(ValueError, match=f"dtype.*{dtype}"):
        open_packed_embedding_dataset(dataset_dir)


def test_write_packed_embedding_dataset_rejects_duplicate_sequence_ids(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "packed"
    embeddings = np.zeros((2, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate sequence IDs"):
        write_packed_embedding_dataset(
            dataset_dir,
            ["seq0", "seq0"],
            embeddings,
            pooling_type="mean",
        )


def test_migrate_npz_directory_to_packed_dataset_uses_final_layer(
    tmp_path: Path,
) -> None:
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    np.savez(
        npz_dir / "seq_b.npz",
        mean_variant_0=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        mean_variant_1=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
    )
    np.savez(
        npz_dir / "seq_a.npz",
        mean_variant_0=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        mean_variant_1=np.asarray([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32),
    )

    output_dir = tmp_path / "packed"
    migrate_npz_directory_to_packed_dataset(
        npz_dir=npz_dir,
        output_dir=output_dir,
        pooling_type="mean",
    )

    embeddings, sequence_ids, metadata = open_packed_embedding_dataset(output_dir)

    assert sequence_ids == ["seq_a", "seq_b"]
    np.testing.assert_array_equal(
        embeddings,
        np.asarray(
            [
                [[7.0, 8.0], [70.0, 80.0]],
                [[3.0, 4.0], [30.0, 40.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert metadata["original_variant_index"] == 0


def test_migrate_npz_directory_to_packed_dataset_sorts_variant_keys_numerically(
    tmp_path: Path,
) -> None:
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    np.savez(
        npz_dir / "seq0.npz",
        mean_variant_10=np.asarray(
            [[100.0, 1000.0], [110.0, 1110.0]],
            dtype=np.float32,
        ),
        mean_variant_2=np.asarray([[20.0, 200.0], [22.0, 222.0]], dtype=np.float32),
    )

    output_dir = tmp_path / "packed"
    migrate_npz_directory_to_packed_dataset(
        npz_dir=npz_dir,
        output_dir=output_dir,
        pooling_type="mean",
    )

    embeddings, _, _ = open_packed_embedding_dataset(output_dir)

    np.testing.assert_array_equal(
        embeddings,
        np.asarray([[[22.0, 222.0], [110.0, 1110.0]]], dtype=np.float32),
    )


def test_migrate_npz_directory_to_packed_dataset_rejects_invalid_original_variant_index(
    tmp_path: Path,
) -> None:
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    np.savez(
        npz_dir / "seq0.npz",
        mean_variant_0=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        mean_variant_1=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="original_variant_index"):
        migrate_npz_directory_to_packed_dataset(
            npz_dir=npz_dir,
            output_dir=tmp_path / "packed",
            pooling_type="mean",
            original_variant_index=2,
        )
