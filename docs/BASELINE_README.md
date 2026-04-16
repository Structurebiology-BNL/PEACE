# Baseline BCE Pipeline

See also:

- [../README.md](../README.md)
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
- [BASELINE_ANALYSIS_README.md](BASELINE_ANALYSIS_README.md)

This repo retains one supervised baseline: `SimplePredictor` trained with binary cross-entropy on last-layer ProtT5 embeddings.

## Supported path

- Model: `SimplePredictor`
- Config: `src/configs/baseline_bce.yaml`
- Training: `uv run effector-bincls train-baseline --config src/configs/baseline_bce.yaml`
- Evaluation: `uv run effector-bincls evaluate-baseline --run_dir <run_dir> --test_csv <csv>`
- Analysis: `uv run effector-bincls analyze-baseline --run_dir <run_dir>`

## Configuration

The retained baseline config uses:

```yaml
data:
  embedding_dir: src/data/embeddings/prott5_dropout_variants_packed

model:
  type: simple_predictor
  input_dim: 1024
  output_dim: 1
  dropout_rate: 0.2
  use_contrastive: false
  encoder_hidden_dim: 512

features:
  normalize: true
  pooling_type: mean
```

`data.embedding_dir` must point to a packed embedding dataset directory with `embeddings.npy`, `sequence_ids.txt`, and `metadata.json`. `pooling_type` must match the dataset metadata. Packed datasets are already final-layer pooled during extraction or legacy conversion, so runtime feature configs do not select transformer layers. Variants are disabled in this path.

Legacy mmap-specific config fields such as `use_mmap` and `mmap_dir` are not part of the supported baseline path. Some older saved configs may still include them as historical artifacts, but the current runtime path ignores them and the public packed-embedding contract does not use them.

## Training flow

1. `effector_bincls.training.baseline` validates `model.type == simple_predictor`.
2. `effector_bincls.training.data` builds `SimpleDataset` folds.
3. `effector_bincls.training.cross_validation` trains one `SimplePredictor` per fold.
4. `effector_bincls.training.trainers.BaselineTrainer` applies BCE loss and threshold selection.

Outputs are written under `results/.../simple_predictor/run_<timestamp>_seed<seed>/` and include `config.yml`, `results.yaml`, fold checkpoints when enabled, and `oof_predictions.npz`.

## Test evaluation

`effector-bincls evaluate-baseline`:

1. Loads pooled out-of-fold predictions.
2. Finds a global threshold.
3. Loads each fold checkpoint.
4. Runs ensemble test inference with simple averaging.

Example:

```bash
uv run effector-bincls evaluate-baseline \
  --run_dir results/baseline_bce/simple_predictor/run_<timestamp> \
  --test_csv src/data/csv_dataset/fungtion_dataset.csv \
  --threshold_method youden

./scripts/run_validation.sh \
  results/baseline_bce/simple_predictor/run_<timestamp> \
  src/data/csv_dataset/fungtion_dataset.csv
```

Expected outputs:

- `test_evaluation.yaml`
- `test_metrics.png`
- `threshold_analysis_oof.png`
- `threshold_analysis_test.png`

## Analysis

```bash
uv run effector-bincls analyze-baseline \
  --run_dir results/baseline_bce/simple_predictor/run_<timestamp>
```

This generates `baseline_analysis/` inside the run directory, including `baseline_analysis_summary.json` and the retained diagnostic plots.

## Unsupported after refactor

- Any removed all-layer architecture path
- Any config or checkpoint that depends on removed all-layer embeddings
