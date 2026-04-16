# Validation Guide (Labeled Data)

This guide covers evaluation on datasets with known labels for the retained baseline and prototype workflows.

See also:

- [../README.md](../README.md)
- [BASELINE_README.md](BASELINE_README.md)
- [PROTOTYPE_RANKING_README.md](PROTOTYPE_RANKING_README.md)

## What Validation Does

Both evaluators follow the same high-level pattern:

1. load pooled out-of-fold predictions from the training run,
2. choose a global threshold,
3. load fold checkpoints,
4. run ensemble predictions on the labeled test CSV,
5. write machine-readable results and diagnostic plots into the run directory.

The default threshold behavior is not a hardcoded `0.5`; it is derived from pooled OOF predictions unless you choose a different threshold method.

## Direct Commands

### Baseline

```bash
uv run effector-bincls evaluate-baseline \
  --run_dir /path/to/baseline_run \
  --test_csv /path/to/dataset.csv \
  --threshold_method youden
```

### Prototype Two-Stage

```bash
uv run effector-bincls evaluate-prototype \
  --run_dir /path/to/prototype_two_stage_run \
  --test_csv /path/to/dataset.csv \
  --threshold_method youden
```

### Prototype Single-Stage

```bash
uv run effector-bincls evaluate-prototype \
  --run_dir /path/to/prototype_single_stage_run \
  --test_csv /path/to/dataset.csv \
  --single-stage \
  --threshold_method youden
```

## Wrapper Script

`scripts/run_validation.sh` is the package-first helper for labeled-data validation:

```bash
./scripts/run_validation.sh <run_dir> <test_csv> [extra_args...]
```

Examples:

```bash
./scripts/run_validation.sh /path/to/baseline_run /path/to/dataset.csv
./scripts/run_validation.sh /path/to/prototype_two_stage_run /path/to/dataset.csv --threshold_method mcc
./scripts/run_validation.sh /path/to/prototype_single_stage_run /path/to/dataset.csv --single-stage
```

Behavior:

- reads `run_dir/config.yml` to detect `model.type`
- routes `simple_predictor` runs to `evaluate-baseline`
- routes `simple` runs to `evaluate-prototype`
- for prototype runs, prefers `fold_*/finetuning/checkpoint.pt` and falls back to `fold_*/checkpoint.pt`
- forwards extra flags such as `--threshold_method` and `--target_recall`

## Common Arguments

- `--run_dir`: saved training run directory
- `--test_csv`: labeled CSV with at least `sequence_id`, `label`, and `partition`
- `--threshold_method`: one of `youden`, `f1`, `mcc`, or `recall_constrained`
- `--target_recall`: used only with `recall_constrained`
- `--single-stage`: required for direct single-stage prototype evaluation and accepted by the wrapper as an override

## Expected Outputs

Validation writes into the run directory:

- `test_evaluation.yaml`
- `test_metrics.png`
- `threshold_analysis_oof.png`
- `threshold_analysis_test.png`
- matching `.csv` summaries for threshold analysis

Prototype validation also records the detected training type in `test_evaluation.yaml`.

## Notes

- Use the validation wrapper for labeled data and the inference wrapper for unlabeled embeddings.
- Historical `model.type: simple` prototype compatibility remains inference-only.
- If the saved run config points to different label column names, wire them through `data.label_config`.
