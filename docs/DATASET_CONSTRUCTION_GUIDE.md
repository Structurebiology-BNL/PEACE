# Dataset Construction Guide

See also:

- [../README.md](../README.md)
- [../src/data/dataset_construction/README.md](../src/data/dataset_construction/README.md)

This guide restores the dataset-design and technical pipeline detail for the
effector benchmark construction workflow. The short README inside
`src/data/dataset_construction/` remains the provenance manifest; this document
is the deeper explanation of why the pipeline exists and how its artifacts map
to the tracked CSVs.

## Scope

This guide applies to the tracked `effector_*` benchmark datasets:

- `src/data/csv_dataset/effector_dataset.csv`
- `src/data/csv_dataset/effector_pretrain_dataset.csv`
- `src/data/csv_dataset/effector_finetune_dataset.csv`

It does **not** describe the origin of `src/data/csv_dataset/fungtion_dataset.csv`,
which is a separate retained benchmark and the default runtime dataset in the
main configs.

## Design goals

The construction pipeline was designed to optimize for four constraints:

1. Reuse as many literature-backed positive sequences as possible.
2. Prevent train/test leakage from close homologs.
3. Reduce label noise in the negative class.
4. Produce datasets that support both representation learning and supervised
   evaluation under heavy class imbalance.

That is why the pipeline is more than a simple concatenate-and-split process.
It intentionally adds clustering, cluster-aware partitioning, BLAST-based
filtering, and asymmetric dataset outputs for pretraining versus finetuning.

## Data sources

### Positive sequences

The positive pool combines curated or literature-derived effector sets from:

- EffectorP 1.0
- EffectorP 2.0
- EffectorP 3.0
- POOE
- Fungtion
- Predector
- EffHunter
- WideEffHunter
- FunEffector-Pred

These FASTA files remain under `src/data/dataset_construction/positive_seqs/`.

### Negative sequences

The negative pool is described in the historical workflow as coming from:

- curated non-effector proteins associated with EffectorP resources
- additional synthetic or tool-derived non-effectors used to increase coverage

The full raw negative-source inputs are not all retained in-tree, which is why
the representative negative CSVs are now treated as important provenance
artifacts rather than disposable scratch outputs.

## Pipeline overview

The construction flow has three logical phases.

### 1. Positive aggregation and de-duplication

The first step concatenates all positive FASTA files and removes exact sequence
duplicates with `seqkit rmdup`.

Purpose:

- collapse exact duplicates across source datasets
- prevent inflated positive counts from duplicated literature entries
- create a stable positive pool before similarity-aware partitioning

Important output:

- `combined_positives_deduplicated.fasta`

### 2. Hierarchical positive clustering

Positive sequences are clustered progressively:

- CD-HIT at 90%
- CD-HIT at 75%
- CD-HIT at 60%
- PSI-CD-HIT at 50%
- PSI-CD-HIT at 40%

The cluster files are then merged with `clstr_rev.pl` into the final
`combined90-75-60-50-40.clstr`.

Purpose:

- group homologous positives before splitting
- prevent close relatives from landing in both train and test
- preserve a broader set of positives for representation learning while still
  choosing one representative per cluster for supervised train/test partitions

Key parameters that matter:

- identity thresholds down to 40%
- coverage thresholds of roughly 80% during clustering
- global alignment mode for the retained commands

The 40% endpoint is the critical design choice. It is permissive enough to keep
the positive class biologically diverse, but strict enough to make the test set
meaningfully separated from the training clusters.

### 3. Cluster-based positive partitioning

`positive_data_partition.py` uses the final cluster file plus the deduplicated
FASTA to produce `combined_positives.csv`.

The output intentionally has three partitions:

- `test`: one representative sequence from held-out clusters
- `train`: one representative sequence from training clusters
- `pretrain`: all sequences from the training clusters, including the
  representatives

This is a deliberate two-objective design:

- supervised train/test evaluation uses cluster representatives
- representation learning can still see the full diversity of the training-side
  clusters

The positive partitioning step also performs a BLAST-based redundancy check so
that test positives do not remain too close to the `pretrain` pool.

## Negative design

The negative path has two distinct goals:

1. reduce redundancy within the negative pool
2. reduce ambiguity between negatives and positives

### Representative extraction

`process_cdhit_clusters.py` converts clustered negative sequences into a
representative CSV:

- `new_negative_representatives.csv`

This prevents the negative class from being dominated by many near-identical
proteins.

### Similarity-based negative filtering

`filter_negatives_by_similarity.py` compares the representative negative pool
against the positive set and removes negatives that are too similar.

Retained thresholds in the documented flow:

- identity threshold: 40%
- coverage threshold: 60%
- BLAST E-value threshold: `1e-5`

The design rationale is conservative: a borderline negative is more dangerous
than a missing negative because it can inject label noise directly into both
training and reported evaluation.

Important output:

- `filtered_new_negative_representatives.csv`

## Final dataset assembly

`combine_pos_and_neg_csv.py` creates the tracked effector datasets from:

- `combined_positives.csv`
- `filtered_new_negative_representatives.csv`

### Why there are three tracked CSVs

#### `effector_dataset.csv`

This is the full construction snapshot. It keeps:

- positive `train`, `test`, and `pretrain`
- negative `train`, `test`, and `pretrain`

Treat it as provenance-first. It is not the main documented runtime input.

#### `effector_pretrain_dataset.csv`

This is the representation-learning view of the data:

- includes original `train` and `pretrain`
- relabels both partitions to `train`
- excludes `test`

Design consequence:

- pretraining sees the broadest retained sequence diversity
- negative:positive ratio is at least the configured target and may be higher

#### `effector_finetune_dataset.csv`

This is the supervised training/evaluation view:

- keeps positive `train`
- keeps all `test`
- keeps negatives allocated to `train` and `test`

Design consequence:

- the supervised splits have the exact requested negative:positive ratio
- evaluation happens on a held-out test partition rather than on the broader
  `pretrain` pool

## Class-imbalance design

The retained combine step uses `--negative-ratio 50`, which means:

- train negatives are sampled to 50x the positive count
- test negatives are sampled to 50x the positive count

This is not arbitrary. The benchmark is intended to reflect the operational
setting where effectors are rare within larger secretome-like candidate sets.
The pipeline therefore prioritizes realistic class imbalance over convenience.

## Leakage prevention strategy

The dataset design uses multiple safeguards rather than relying on one split:

- exact de-duplication removes duplicate positives
- cluster-based splitting removes near-homolog leakage across train/test
- BLAST-based positive/test versus pretrain filtering adds a second safety layer
- BLAST-based negative filtering removes ambiguous negatives

This layered design matters because a simple random split would dramatically
overestimate performance in a protein-family setting.

## Retained provenance artifacts

The provenance manifest keeps the smallest artifact set that still explains or
reconstructs the tracked effector benchmark CSVs:

- `combined_positives.csv`
- `combined_positives_deduplicated.fasta`
- `combined90-75-60-50-40.clstr`
- `new_negative_representatives.csv`
- `filtered_new_negative_representatives.csv`

Archived intermediates such as `combined_positives.fasta`, `combined_40.fasta`,
and `combined_40.clstr` are now moved under ignored `backup/` because they are
useful for ad hoc reconstruction but not required to explain the tracked CSVs.

## Reconstruction entrypoints

For current reproduction from the retained artifacts:

1. `./scripts/benchmarking/rebuild_positive_sequence_provenance.sh`
2. `uv run python -m src.data.dataset_construction.positive_data_partition ...`
3. `uv run python -m src.data.dataset_construction.process_cdhit_clusters ...`
4. `uv run python -m src.data.dataset_construction.filter_negatives_by_similarity ...`
5. `uv run python -m src.data.dataset_construction.combine_pos_and_neg_csv ...`

The exact command forms are listed in
`src/data/dataset_construction/README.md`.

## Practical interpretation

If you are working on runtime training or evaluation code:

- use `fungtion_dataset.csv`, `effector_pretrain_dataset.csv`, or
  `effector_finetune_dataset.csv` depending the workflow
- do not treat the construction artifacts as direct runtime dependencies

If you are working on dataset provenance or release packaging:

- treat the retained construction CSVs and cluster/FASTA artifacts as auditable
  evidence for the tracked effector benchmark
- treat `effector_dataset.csv` as provenance-only unless you are explicitly
  re-running the construction logic

