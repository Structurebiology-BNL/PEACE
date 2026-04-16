import json
from pathlib import Path

import numpy as np
import torch

from effector_bincls import notebook_support


def _notebook_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    notebook_path = repo_root / "notebooks" / "fungus_inference_colab.ipynb"
    notebook = json.loads(notebook_path.read_text())
    return "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )


def _notebook_support_source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    support_path = repo_root / "src" / "effector_bincls" / "notebook_support.py"
    return support_path.read_text()


def test_notebook_does_not_import_legacy_src_data_modules() -> None:
    source = _notebook_source()

    assert "src.data." not in source
    assert "from effector_bincls.notebook_support import" in source


def test_notebook_reads_bundled_model_metadata() -> None:
    source = _notebook_source()

    assert "pretrained_models/fungus_model" in source
    assert "load_notebook_model_metadata(MODEL_DIR)" in source
    assert "metadata.yml" in source
    assert "DEFAULT_THRESHOLD = 0.4641951620578766" not in source


def test_notebook_uses_packed_embeddings_without_layer_idx_or_npz() -> None:
    source = _notebook_source()
    notebook_support_source = _notebook_support_source()

    assert "extract_prott5_embeddings(" in source
    assert "from effector_bincls.data import open_packed_embedding_dataset" in source
    assert "open_packed_embedding_dataset(" in source
    assert "run_inference(" in source
    assert "metadata.json" in source
    assert "embeddings.npy" in source
    assert "write_packed_embedding_dataset" in notebook_support_source
    assert "write_packed_embedding_dataset(" in notebook_support_source
    assert "layer_idx" not in source
    assert ".npz" not in source
    assert "glob('*.npz')" not in source


def test_extract_prott5_embeddings_writes_packed_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeParameter:
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    class _FakeModel:
        def __init__(self) -> None:
            self.training = True

        def parameters(self):
            yield _FakeParameter()

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

    final_hidden_states = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    monkeypatch.setattr(
        notebook_support,
        "load_prott5_model",
        lambda device_str="auto": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        notebook_support,
        "_encode_sequence",
        lambda tokenizer, model, sequence: final_hidden_states,
    )
    monkeypatch.setattr(notebook_support.torch.cuda, "is_available", lambda: False)

    output_dir, device = notebook_support.extract_prott5_embeddings(
        [{"safe_id": "seq1", "sequence": "ACDE"}],
        tmp_path / "packed_embeddings",
        pooling_type="mean",
        num_variants=1,
        device_str="cpu",
    )

    assert output_dir == tmp_path / "packed_embeddings"
    assert device == torch.device("cpu")
    assert (output_dir / "embeddings.npy").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "sequence_ids.txt").read_text() == "seq1\n"
    assert list(output_dir.glob("*.npz")) == []

    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape == (1, 1, 2)
    np.testing.assert_allclose(embeddings[0, 0], np.array([6.0, 7.0], dtype=np.float32))

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["pooling_type"] == "mean"
    assert metadata["num_sequences"] == 1
    assert metadata["num_variants"] == 1
    assert metadata["embedding_dim"] == 2
    assert metadata["original_variant_index"] == 0


def test_extract_prott5_embeddings_writes_packed_dataset_with_variants(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeParameter:
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    class _FakeModel:
        def __init__(self) -> None:
            self.training = True

        def parameters(self):
            yield _FakeParameter()

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

    deterministic_hidden_states = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    variant_hidden_states = [
        torch.tensor([[15.0, 16.0], [17.0, 18.0]]),
        torch.tensor([[25.0, 26.0], [27.0, 28.0]]),
    ]
    variant_iter = iter(variant_hidden_states)

    def _fake_encode_sequence(tokenizer, model, sequence):
        if model.training:
            return next(variant_iter)
        return deterministic_hidden_states

    monkeypatch.setattr(
        notebook_support,
        "load_prott5_model",
        lambda device_str="auto": (object(), _FakeModel()),
    )
    monkeypatch.setattr(notebook_support, "_encode_sequence", _fake_encode_sequence)
    monkeypatch.setattr(notebook_support.torch.cuda, "is_available", lambda: False)

    output_dir, device = notebook_support.extract_prott5_embeddings(
        [{"safe_id": "seq1", "sequence": "ACDE"}],
        tmp_path / "packed_embeddings_variants",
        pooling_type="mean",
        num_variants=3,
        device_str="cpu",
    )

    assert output_dir == tmp_path / "packed_embeddings_variants"
    assert device == torch.device("cpu")
    assert (output_dir / "embeddings.npy").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "sequence_ids.txt").read_text() == "seq1\n"
    assert list(output_dir.glob("*.npz")) == []

    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape == (1, 3, 2)
    np.testing.assert_allclose(
        embeddings[0],
        np.array(
            [
                [6.0, 7.0],
                [16.0, 17.0],
                [26.0, 27.0],
            ],
            dtype=np.float32,
        ),
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["pooling_type"] == "mean"
    assert metadata["num_sequences"] == 1
    assert metadata["num_variants"] == 3
    assert metadata["embedding_dim"] == 2
    assert metadata["original_variant_index"] == 0


def test_extract_prott5_embeddings_uses_true_final_hf_layer(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeParameter:
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    class _FakeTokenizer:
        def __call__(self, sequences, add_special_tokens=True, padding="longest"):
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    class _FakeOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class _FakeModel:
        def __init__(self) -> None:
            self.training = True

        def parameters(self):
            yield _FakeParameter()

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

        def __call__(self, input_ids, attention_mask, output_hidden_states=True):
            # HF-style tuple: embeddings + successive encoder layers.
            return _FakeOutput(
                (
                    torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
                    torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                    torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
                    torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),
                )
            )

    monkeypatch.setattr(
        notebook_support,
        "load_prott5_model",
        lambda device_str="auto": (_FakeTokenizer(), _FakeModel()),
    )
    monkeypatch.setattr(notebook_support.torch.cuda, "is_available", lambda: False)

    output_dir, _ = notebook_support.extract_prott5_embeddings(
        [{"safe_id": "seq1", "sequence": "ACDE"}],
        tmp_path / "packed_embeddings_final_layer",
        pooling_type="mean",
        num_variants=1,
        device_str="cpu",
    )

    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape == (1, 1, 2)
    np.testing.assert_allclose(
        embeddings[0, 0],
        np.array([10.0, 11.0], dtype=np.float32),
    )
