# Baseline Analysis Guide

See also:

- [../README.md](../README.md)
- [BASELINE_README.md](BASELINE_README.md)
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)

This guide covers baseline model analysis for the retained `SimplePredictor` BCE workflow.

## Purpose

`effector-bincls analyze-baseline` provides:
- Embedding-space visualization (UMAP with PCA fallback).
- Feature-space diagnostics (norms and class separation).
- Threshold and classification-metric analysis.
- Saved analysis artifacts for reporting.

## Run Analysis

```bash
uv run effector-bincls analyze-baseline \
  --run_dir /path/to/baseline_run
```

Optional arguments:

```bash
uv run effector-bincls analyze-baseline \
  --run_dir /path/to/baseline_run \
  --fold 1 \
  --sample_size 200
```

## Required Run Artifacts

The run directory should contain at least:
- `config.yml`
- `oof_predictions.npz`
- `fold_X/checkpoint.pt`

## Outputs

Generated under `baseline_analysis/` inside the run directory:
- `embedding_space_visualization.png`
- `feature_space_analysis.png`
- `threshold_summary.png`
- `baseline_analysis_summary.json`
- `baseline_analysis_report.txt`

## Troubleshooting

- Missing checkpoints: verify fold checkpoint files exist.
- Large-memory runs: reduce `--sample_size`.
- UMAP instability: analysis falls back to PCA automatically.
