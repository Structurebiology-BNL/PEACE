# Dataset Construction Provenance

This directory is a provenance area for the tracked effector benchmark CSVs.
The active package workflows do not read these files directly during normal
training or evaluation. They matter because they explain how the tracked
`effector_*` datasets were assembled.

For the full dataset-design rationale, data-source description, clustering
strategy, negative filtering logic, and class-imbalance discussion, see
`docs/DATASET_CONSTRUCTION_GUIDE.md`.

## Runtime vs provenance

Supported runtime datasets:

- `src/data/csv_dataset/fungtion_dataset.csv`
- `src/data/csv_dataset/effector_pretrain_dataset.csv`
- `src/data/csv_dataset/effector_finetune_dataset.csv`

Provenance-only dataset:

- `src/data/csv_dataset/effector_dataset.csv`

Important boundary:

- The default retained configs still point at `fungtion_dataset.csv`.
- The alternate simple prototype config points at `effector_finetune_dataset.csv`.
- The files in this directory are not part of the default runtime path.
- `fungtion_dataset.csv` is not regenerated from the artifacts in this directory.

## Retained provenance artifacts

Keep these files in the repo because they are the minimal local explanation for
the tracked `effector_*` datasets:

- `combined_positives.csv`
- `combined_positives_deduplicated.fasta`
- `combined90-75-60-50-40.clstr`
- `new_negative_representatives.csv`
- `filtered_new_negative_representatives.csv`

Tracked source inputs that remain useful:

- `positive_seqs/*.fasta`
- `positive_data_partition.py`
- `process_cdhit_clusters.py`
- `filter_negatives_by_similarity.py`
- `combine_pos_and_neg_csv.py`
- `utils.py`

## Archived intermediates

These files are useful for ad hoc reconstruction but are not part of the
minimal retained provenance set. They should live under ignored `backup/`
rather than cluttering the main tree:

- `combined_positives.fasta`
- `combined_40.fasta`
- `combined_40.clstr`
- the old host-specific `positive_seqs/sequence_processing.sh`

Use the sanitized helper at
`scripts/benchmarking/rebuild_positive_sequence_provenance.sh` instead of the
old shell script.

## Provenance relationships

The retained artifacts line up with the tracked CSVs as follows:

- `combined_positives.csv` explains all positive rows in
  `src/data/csv_dataset/effector_dataset.csv`
- `filtered_new_negative_representatives.csv` explains all negative rows in
  `src/data/csv_dataset/effector_dataset.csv`
- `effector_pretrain_dataset.csv` is the `train` plus `pretrain` portion of
  `effector_dataset.csv`, with `partition` relabeled to `train`
- `effector_finetune_dataset.csv` is the `train` positives plus all `train` and
  `test` negatives plus all `test` rows from `effector_dataset.csv`

The unit tests under `tests/unit/data/` enforce those relationships.

## Reconstruction flow

1. Rebuild positive-sequence clustering intermediates:

```bash
./scripts/benchmarking/rebuild_positive_sequence_provenance.sh
```

2. Regenerate the positive partition CSV:

```bash
uv run python -m src.data.dataset_construction.positive_data_partition \
  --cluster-file src/data/dataset_construction/combined90-75-60-50-40.clstr \
  --fasta-file src/data/dataset_construction/combined_positives_deduplicated.fasta \
  --output-csv src/data/dataset_construction/combined_positives.csv \
  --test-ratio 0.2 \
  --random-seed 42 \
  --label 1 \
  --identity-threshold 40.0 \
  --coverage-threshold 60.0
```

3. Regenerate negative representatives after external clustering:

```bash
uv run python -m src.data.dataset_construction.process_cdhit_clusters \
  --cluster-file <negative_clusters.clstr> \
  --fasta-file <negative_sequences.fasta> \
  --output-csv src/data/dataset_construction/new_negative_representatives.csv \
  --label 0
```

4. Filter negatives against the positive set:

```bash
uv run python -m src.data.dataset_construction.filter_negatives_by_similarity \
  --mode separate \
  --positives-csv src/data/dataset_construction/combined_positives.csv \
  --negatives-csv src/data/dataset_construction/new_negative_representatives.csv \
  --output-csv src/data/dataset_construction/filtered_new_negative_representatives.csv \
  --identity-threshold 40.0 \
  --coverage-threshold 60.0 \
  --evalue-threshold 1e-5
```

5. Regenerate the tracked effector CSV variants:

```bash
uv run python -m src.data.dataset_construction.combine_pos_and_neg_csv \
  --positive-csv src/data/dataset_construction/combined_positives.csv \
  --negative-csv src/data/dataset_construction/filtered_new_negative_representatives.csv \
  --pretrain-csv src/data/csv_dataset/effector_pretrain_dataset.csv \
  --finetune-csv src/data/csv_dataset/effector_finetune_dataset.csv \
  --negative-ratio 50 \
  --random-seed 42
```

`effector_dataset.csv` remains the full combined snapshot with `train`,
`test`, and `pretrain` partitions.
