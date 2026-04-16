# Prototype Inference Guide

See also:

- [../README.md](../README.md)
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
- [PROTOTYPE_RANKING_README.md](PROTOTYPE_RANKING_README.md)

This guide covers inference for prototype contrastive models that use the retained MLP architecture.

The repo also ships a supported low-code Colab path at `notebooks/fungus_inference_colab.ipynb`, backed by the bundled model in `pretrained_models/fungus_model/`. The notebook reads its default threshold from `pretrained_models/fungus_model/metadata.yml`.

## What It Does

`effector-bincls infer-prototype`:
1. Loads a packed embedding dataset for unseen sequences.
2. Loads fold checkpoints from a trained run.
3. Runs fold-wise predictions and averages them.
4. Applies a binary threshold.
5. Writes `sequence_id, probability, binary_prediction, threshold` to CSV.

## Inputs

- Packed embedding dataset directory.
- Trained run directory with `config.yml` and fold checkpoints.
- Supported checkpoint layouts:
  - Two-stage (default): `fold_X/finetuning/checkpoint.pt`
  - Single-stage: `fold_X/checkpoint.pt` with `--single-stage`

## Packed Embedding Dataset Requirements

`--embedding_dir` must point to a packed dataset directory containing:

- `embeddings.npy`
- `sequence_ids.txt`
- `metadata.json`

The array in `embeddings.npy` must have shape `[num_sequences, num_variants, embedding_dim]`. `sequence_ids.txt` must list one sequence ID per line in the same row order as the array. `metadata.json` must describe the packed dataset, including:

- `format_version`
- `layout`
- `num_sequences`
- `num_variants`
- `embedding_dim`
- `pooling_type`
- `original_variant_index`
- `dtype`

`--pooling_type` must match `metadata.json`. Variant handling is determined by the saved run config loaded from `model_dir/config.yml`, not by a dedicated inference CLI switch. When the saved run config does not use variants, inference uses the packed dataset's `original_variant_index` to select the canonical embedding for each sequence.

## Run Inference

### Public notebook path

For low-code fungi-only inference, open `notebooks/fungus_inference_colab.ipynb` in Colab. The notebook clones the repo, installs the package, generates ProtT5 embeddings, and runs inference with the bundled fungi-only model.

### CLI and helper script path


Using helper script:

```bash
./scripts/run_inference.sh /path/to/packed_embeddings /path/to/run_dir
./scripts/run_inference.sh /path/to/packed_embeddings /path/to/run_dir 0.65
./scripts/run_inference.sh /path/to/packed_embeddings /path/to/run_dir 0.5 --single-stage
```

Direct command:

```bash
uv run effector-bincls infer-prototype \
  --embedding_dir /path/to/packed_embeddings \
  --model_dir /path/to/run_dir \
  --threshold 0.5 \
  --output_file /path/to/output/predictions.csv
```

## CLI Arguments

- `--embedding_dir` required. This must be a packed embedding dataset directory.
- `--model_dir` required.
- `--output_file` optional (default: `{model_dir}/predictions.csv`).
- `--threshold` optional (default: `0.5`).
- `--single-stage` optional (default: two-stage checkpoint layout).
- `--batch_size` optional (default: `32`).
- `--pooling_type` optional (`mean`, `max`, `bos`, `eos`).

## Output CSV

```csv
sequence_id,probability,binary_prediction,threshold
seq001,0.823456,1,0.5
seq002,0.234567,0,0.5
```

## Notes

- Inference enforces the retained `model.type: simple` prototype model contract.
- The CLI no longer accepts per-sequence `.npz` inputs or any legacy mmap-specific CLI/config contract.
- Some bundled or historical run artifacts may still contain legacy fields such as `use_mmap` or `mmap_dir` in saved configs. The current packed-embedding runtime path ignores those fields; the supported public contract is the packed dataset directory plus the active runtime settings.
- Historical `model.type: simple` run directories remain supported for inference.
- Use `scripts/run_validation.sh` or `evaluate-prototype` for labeled-data validation.
- Threshold selection should match your validation/test strategy when possible.
