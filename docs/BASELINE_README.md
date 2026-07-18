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

## Contrastive-BCE workflow

The public Contrastive-BCE workflow uses the same `SimplePredictor` encoder and
binary classification head, plus a contrastive projection head:

```bash
uv run effector-bincls train-contrastive-bce \
  --config src/configs/contrastive_bce.yaml
```

For each protein, BCE is applied to the classification logit from the canonical
packed embedding. Dropout-view InfoNCE is applied to all sampled variants. This
is a single-stage comparator: it does not use prototypes, prototype alignment,
prototype-distance scoring, or PEACE's two-stage optimization.

The config requires a packed embedding dataset with at least two variants,
`training.variant_sampling.always_include_original: true`, and a positive
finite contrastive temperature and loss weights. The public config is
deterministic with `hardware.random_seed: 42` and
`hardware.deterministic: true`.

Before a run directory is created, `train-contrastive-bce` validates the complete
configuration, runtime CSV contract, required train/test partitions, both training
classes, per-class support for stratified folds, packed pooling and
dimensionality, requested variant count, canonical variant metadata, and embedding
coverage for every train/test ID. Invalid or ambiguous input fails without leaving
a partial run directory.

Contrastive-BCE produces baseline-compatible fold checkpoints, pooled OOF
predictions, and thresholds. Evaluate a saved run with:

```bash
uv run effector-bincls evaluate-baseline \
  --run_dir results/contrastive_bce/simple_predictor/run_<timestamp> \
  --test_csv src/data/csv_dataset/fungtion_dataset.csv \
  --threshold_method youden
```

InfoNCE constructs a square similarity matrix over
`batch_size * num_variants` projected views, so its memory cost is quadratic in
that product. Reduce `training.batch_size` or
`training.variant_sampling.num_variants` if the matrix does not fit in memory;
do not silently switch to single-view training.

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

Baseline evaluation always uses one canonical embedding per sequence, selected by
the packed dataset's `original_variant_index`. A saved training config may enable
variants for Contrastive-BCE regularization, but baseline test probabilities are
not averaged across variants. Prototype evaluation retains its configured variant
behavior.

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
