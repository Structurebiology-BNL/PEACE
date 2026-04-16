# Prototype Contrastive Training

See also:

- [../README.md](../README.md)
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
- [PROTOTYPE_ANALYSIS_README.md](PROTOTYPE_ANALYSIS_README.md)
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

This repo retains prototype-aware contrastive training for supervised binary classification under extreme class imbalance.

## Supported paths

- Single-stage training:
  `uv run effector-bincls train-prototype-single --config src/configs/prototype_single_stage.yaml`
- Two-stage training:
  `uv run effector-bincls train-prototype-two-stage --config src/configs/prototype_two_stage.yaml`
- Evaluation:
  `uv run effector-bincls evaluate-prototype --run_dir <run_dir> --test_csv <csv>`
- Analysis:
  `uv run effector-bincls analyze-prototype --run_dir <run_dir>`
- Inference:
  `uv run effector-bincls infer-prototype --embedding_dir <packed_dataset_dir> --model_dir <run_dir>`

The public config set is method-oriented. The tracked examples are `src/configs/prototype_single_stage.yaml` and `src/configs/prototype_two_stage.yaml`.

For all retained prototype workflows, `data.embedding_dir` must point to a packed embedding dataset directory containing:

- `embeddings.npy`
- `sequence_ids.txt`
- `metadata.json`

The packed array layout is `[num_sequences, num_variants, embedding_dim]`. `features.pooling_type` must match the packed dataset metadata. Legacy mmap-specific config fields such as `use_mmap` and `mmap_dir` are not part of the supported public contract. Some older saved configs may still carry those fields as historical artifacts, but the current runtime path ignores them.

## Retained model

Both flows use `SimplePredictor` with:

- final-layer pooled embeddings from the packed dataset
- shared encoder
- contrastive embedding head
- prototype-based classification at train and eval time

`model.type` must be `simple`.

## Retained loss behavior

The retained training path keeps the existing loss stack:

- contrastive learning
- prototype alignment
- distance-based classification
- BCE term

The refactor removes only the deleted all-layer transformer path. The retained loss math now lives under `effector_bincls.training.losses`, and the prototype scoring utilities stay package-owned under `effector_bincls.prototype_scoring`.

## Single-stage flow

Use when you want one training stage with the custom loss:

```bash
uv run effector-bincls train-prototype-single \
  --config src/configs/prototype_single_stage.yaml
```

Key components:

- `effector_bincls.training.prototype_single`
- `effector_bincls.training.data`
- `effector_bincls.training.cross_validation`
- `effector_bincls.training.trainers.PrototypeRankingTrainer`

## Two-stage flow

Use when you want prototype-alignment pretraining followed by prototype-ranking fine-tuning:

```bash
uv run effector-bincls train-prototype-two-stage \
  --config src/configs/prototype_two_stage.yaml
```

Key components:

- `effector_bincls.training.prototype_two_stage`
- `effector_bincls.training.data`
- `effector_bincls.training.cross_validation`
- `effector_bincls.training.trainers.PretrainTrainer`
- `effector_bincls.training.trainers.PrototypeRankingTrainer`

Two-stage runs save checkpoints under:

- `fold_<n>/pretraining/checkpoint.pt`
- `fold_<n>/finetuning/checkpoint.pt`

## Evaluation

`effector-bincls evaluate-prototype` evaluates both retained flows.

Single-stage example:

```bash
uv run effector-bincls evaluate-prototype \
  --single-stage \
  --run_dir <single_stage_run_dir> \
  --test_csv src/data/csv_dataset/effector_finetune_dataset.csv \
  --threshold_method youden
```

Two-stage example:

```bash
uv run effector-bincls evaluate-prototype \
  --run_dir <two_stage_run_dir> \
  --test_csv src/data/csv_dataset/fungtion_dataset.csv \
  --threshold_method youden

./scripts/run_validation.sh <two_stage_run_dir> src/data/csv_dataset/fungtion_dataset.csv
```

The evaluator:

1. loads pooled out-of-fold predictions,
2. chooses a global threshold,
3. loads fold checkpoints,
4. computes prototype-distance probabilities,
5. averages fold predictions.

For single-stage runs, either pass `--single-stage` directly or use:

```bash
./scripts/run_validation.sh <single_stage_run_dir> src/data/csv_dataset/effector_finetune_dataset.csv --single-stage
```

## Analysis

```bash
uv run effector-bincls analyze-prototype \
  --run_dir <run_dir> \
  --sample_size 500
```

Use `--single-stage` for single-stage run layouts.

## Compatibility

- Replace any removed all-layer model setting with `model.type: simple`.
- Remove `model.num_layers`, `model.n_head`, and `model.num_transformer_layers`.
- Keep `contrastive_dim` and `encoder_hidden_dim`.
- Historical `model.type: simple` prototype run directories remain supported for inference.
- Old training, evaluation, and analysis entrypoints are not retained.
