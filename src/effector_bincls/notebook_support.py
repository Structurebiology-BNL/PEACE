"""Helpers for the public fungi-only notebook workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from effector_bincls.data import (
    create_packed_embedding_memmap,
    write_packed_embedding_dataset,
)
from effector_bincls.run_utils import load_config

PROTT5_MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"


def _validate_device(device_str: str) -> torch.device:
    """Validate a notebook-facing device specification."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str.startswith("cuda:"):
        try:
            gpu_id = int(device_str.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid CUDA device specification: {device_str}"
            ) from exc

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system")
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                "GPU "
                f"{gpu_id} not found. Available GPUs: 0 to "
                f"{torch.cuda.device_count() - 1}"
            )
        return torch.device(device_str)

    raise ValueError(
        f"Invalid device specification: {device_str}. Use 'cpu', 'cuda:N', or 'auto'."
    )


def load_prott5_model(device_str: str = "auto") -> tuple[T5Tokenizer, T5EncoderModel]:
    """Load the ProtT5 encoder for public notebook use."""
    device = _validate_device(device_str)

    try:
        tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL_NAME, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(PROTT5_MODEL_NAME).to(device)
    except Exception as exc:  # pragma: no cover - depends on model download/runtime
        raise RuntimeError(
            f"Failed to load ProtT5-XL-UniRef50 on device {device}: {exc}"
        ) from exc

    if device.type == "cpu":
        model = model.to(torch.float32)
    else:
        model = model.half()

    return tokenizer, model


def pool_prott5_hidden_states(
    hidden_states: torch.Tensor | Sequence[torch.Tensor] | np.ndarray,
    *,
    has_bos_token: bool = False,
) -> dict[str, np.ndarray]:
    """Pool ProtT5 hidden states into the supported embedding keys."""
    if isinstance(hidden_states, Sequence) and not isinstance(
        hidden_states, (str, bytes, np.ndarray)
    ):
        hidden_states = [
            state.detach().cpu().float().numpy() for state in hidden_states
        ]
        hidden_states = np.stack(hidden_states)
    elif torch.is_tensor(hidden_states):
        hidden_states = hidden_states.detach().cpu().float().numpy()
    else:
        hidden_states = np.asarray(hidden_states)

    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    if hidden_states.ndim == 2:
        pooled = {
            "mean": hidden_states.mean(axis=0),
            "max": hidden_states.max(axis=0),
            "eos": hidden_states[-1, :],
        }
        if has_bos_token:
            pooled["bos"] = hidden_states[0, :]
        return pooled

    if hidden_states.ndim != 3:
        raise ValueError(
            "Expected hidden states with shape "
            "[seq_len, embed_dim] or [n_layers, seq_len, embed_dim] "
            "after squeezing batch dimensions."
        )

    pooled = {
        "mean": hidden_states.mean(axis=1),
        "max": hidden_states.max(axis=1),
        "eos": hidden_states[:, -1, :],
    }
    if has_bos_token:
        pooled["bos"] = hidden_states[:, 0, :]
    return pooled


def _encode_sequence(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    sequence: str,
) -> torch.Tensor:
    """Run one ProtT5 forward pass and return the final encoder layer."""
    device = next(model.parameters()).device
    spaced_sequence = " ".join(sequence)
    tokenized = tokenizer([spaced_sequence], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tokenized["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)

    with torch.no_grad():
        embedding_repr = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    return embedding_repr.hidden_states[-1].squeeze(0)


def extract_prott5_embeddings(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    pooling_type: str = "mean",
    num_variants: int = 8,
    device_str: str = "auto",
) -> tuple[Path, torch.device]:
    """Extract notebook embeddings and save them as a packed dataset."""
    if pooling_type not in {"mean", "max", "bos", "eos"}:
        raise ValueError("pooling_type must be one of: 'mean', 'max', 'bos', 'eos'")
    if num_variants < 1:
        raise ValueError("num_variants must be at least 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_prott5_model(device_str)
    device = next(model.parameters()).device
    sequence_ids = [str(record["safe_id"]) for record in records]
    writer: np.memmap | None = None

    for row_index, record in enumerate(
        tqdm(records, desc="Extracting ProtT5 embeddings")
    ):
        sequence = str(record["sequence"])

        variant_embeddings: list[np.ndarray] = []
        original_training_mode = model.training

        model.eval()
        deterministic_hidden_states = _encode_sequence(tokenizer, model, sequence)
        deterministic_pooling = pool_prott5_hidden_states(
            deterministic_hidden_states,
            has_bos_token=False,
        )
        if pooling_type not in deterministic_pooling:
            raise KeyError(
                f"Pooling key '{pooling_type}' not found in extracted embeddings. "
                f"Available keys: {list(deterministic_pooling.keys())}"
            )
        variant_embeddings.append(deterministic_pooling[pooling_type])

        if num_variants > 1:
            model.train()
            for _ in range(num_variants - 1):
                dropout_hidden_states = _encode_sequence(tokenizer, model, sequence)
                dropout_pooling = pool_prott5_hidden_states(
                    dropout_hidden_states,
                    has_bos_token=False,
                )
                variant_embeddings.append(dropout_pooling[pooling_type])

        model.train(original_training_mode)
        row = np.stack(variant_embeddings, axis=0).astype(np.float32, copy=False)
        if writer is None:
            writer = create_packed_embedding_memmap(
                output_dir,
                sequence_ids,
                shape=(len(sequence_ids), num_variants, row.shape[1]),
                dtype=row.dtype,
                pooling_type=pooling_type,
                original_variant_index=0,
            )
        writer[row_index] = row

    if writer is None:
        write_packed_embedding_dataset(
            output_dir,
            sequence_ids,
            np.empty((0, num_variants, 0), dtype=np.float32),
            pooling_type=pooling_type,
            original_variant_index=0,
        )
    else:
        writer.flush()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir, device


def load_notebook_model_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Load validated notebook metadata for the bundled public model."""
    metadata_path = Path(model_dir) / "metadata.yml"
    metadata = load_config(metadata_path)
    notebook_metadata = metadata.get("notebook")
    if not isinstance(notebook_metadata, dict):
        raise ValueError(
            f"metadata.yml must contain a 'notebook' mapping: {metadata_path}"
        )
    if "default_threshold" not in notebook_metadata:
        raise ValueError(
            "metadata.yml must define notebook.default_threshold "
            f"for the bundled model: {metadata_path}"
        )
    return metadata


__all__ = [
    "extract_prott5_embeddings",
    "load_notebook_model_metadata",
    "load_prott5_model",
    "pool_prott5_hidden_states",
]
