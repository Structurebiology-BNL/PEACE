# Contrastive-BCE Contract and Evaluation Design

## Status

Approved design decisions from the review of PR #1:

- Enforce labeled runtime CSV contracts globally across public PEACE workflows.
- Materialize the pretraining runtime CSV as the unique union of the provenance
  `train` and `pretrain` memberships while preserving the provenance snapshot.
- Use canonical-only predictions for Contrastive-BCE baseline evaluation.
- Do not require a CUDA determinism or memory run for this change.

## Goal

Make public labeled workflows reject invalid data deterministically, make the
Contrastive-BCE entrypoint complete all input validation before creating a run
directory, and evaluate Contrastive-BCE checkpoints using the same canonical
classification semantics used for BCE training and OOF threshold calibration.

## Non-goals

- Add multi-view probability ensembling or view-based uncertainty outputs.
- Change the Contrastive-BCE training loss or its use of multiple views.
- Change checkpoint, OOF prediction, threshold, or result artifact formats.
- Change any unique sequence membership, label, or train/test allocation.
- Deduplicate the provenance-first construction snapshot, whose repeated IDs
  encode membership in both `train` and `pretrain`.
- Require GPU-specific validation for merge readiness.

## Current Behavior and Problems

The shared labeled-data loader checks only required columns and requested
partitions. It permits duplicate sequence IDs, missing values, and labels
outside the binary domain. Duplicate IDs can enter both sides of a
cross-validation fold, and `SimpleDataset` can silently overwrite a duplicate
label when it constructs its ID-to-label mapping.

The Contrastive-BCE config validator checks selected model and loss fields, but
does not validate every field needed during training. Data and embedding checks
run only after `setup_training` creates and populates a run directory. Invalid
embedding paths, incompatible metadata, or infeasible class counts can
therefore leave partial artifacts or fail after training has begun.

During baseline evaluation, a Contrastive-BCE config causes every sampled view
to pass through the encoder and contrastive head. The classifier uses only the
first view and the evaluator discards the contrastive output, so the extra work
does not affect probabilities.

## Design

### 1. Global labeled runtime CSV contract

`load_labeled_dataset` remains the single enforcement point for the shared
runtime CSV contract. It is not the loader for provenance-first construction
snapshots. Every call must reject:

- null or empty sequence IDs;
- duplicate sequence IDs anywhere in the CSV, including duplicates within one
  partition and reuse across partitions;
- null labels;
- labels that are not numeric, integer-valued `0` or `1`;
- null or empty partition values; and
- missing caller-requested partitions.

Validation must not deduplicate, coerce labels, synthesize partitions, or keep
the first or last conflicting record. Errors must name the source CSV, the
violated contract, and a bounded sample of offending IDs or values.

The loader may continue to accept additional non-empty partition names when a
runtime workflow explicitly requires them. Each workflow remains responsible
for declaring its required partitions. There is no runtime opt-out from ID
uniqueness.

This is an intentional tightening of a public data contract. Existing valid
CSV inputs remain compatible; malformed inputs that previously proceeded are
expected to fail.

### 2. Provenance membership and runtime pretraining materialization

`effector_dataset.csv` remains a provenance-first snapshot rather than a
runtime input. Its repeated IDs are meaningful membership records: the same
547 positive sequences appear once in `train` and once in `pretrain`. Raw
provenance validation must confirm that repeated IDs have identical sequences
and labels and differ only in their permitted membership partitions.

`effector_pretrain_dataset.csv` is a runtime view and must contain one row per
unique sequence. Dataset construction forms the set union of the provenance
`train` and `pretrain` memberships, relabels the resulting rows to runtime
partition `train`, verifies label and sequence consistency for repeated IDs,
and emits one deterministic row per `sequence_id`.

For the tracked artifact this changes the row count from 33,231 to 32,684 by
removing 547 redundant positive rows. It preserves all 32,684 unique sequence
IDs, all labels, the 27,897-ID finetuning train subset, the 4,787 additional
pretraining IDs, and zero overlap with the 7,038-ID test set.

The package two-stage loader already constructs `pretraining_samples` as a
dictionary and `pretraining_ids` as a set before fold creation. Therefore this
materialization change does not alter effective training membership, labels,
or class counts in the current package pipeline. It only makes the tracked
runtime CSV conform to the behavior the loader already applies in memory.

Provenance tests read `effector_dataset.csv` and construction intermediates as
provenance artifacts, not through `load_labeled_dataset`. They separately
verify the permitted membership relationship and the deterministic unique
runtime materialization.

### 3. Complete Contrastive-BCE preflight

Preflight has two explicit layers:

1. `validate_contrastive_bce_config(config)` validates the complete
   Contrastive-BCE configuration schema and value ranges.
2. A dedicated input validator reads the labeled CSV and packed embedding
   metadata and validates their relationship to the config.

Both layers run before `setup_training`. The input validator is read-only and
does not create directories, log files, configs, folds, or checkpoints.

Configuration validation covers all fields required by the public workflow:

- required `data`, `features`, `model`, `training`, `output`, and `hardware`
  sections;
- finite numeric optimizer and scheduler values with valid ranges;
- positive integer batch, epoch, and fold counts;
- supported threshold method and scheduler enums;
- `mode` consistent with the monitored metric;
- `target_recall` in `(0, 1]`;
- the existing Contrastive-BCE requirements for model type, scalar output,
  positive loss weights and temperature, and at least two sampled views; and
- non-negative worker count and a valid CPU/GPU identifier convention.

Input validation covers:

- the global labeled CSV contract;
- presence of `train` and `test` partitions;
- both binary classes in the training partition;
- at least `training.num_folds` samples from each class, ensuring every
  stratified validation fold can compute AUPRC and ROC-AUC;
- a valid packed embedding dataset and metadata;
- `metadata.pooling_type == features.pooling_type`;
- `metadata.embedding_dim == model.input_dim`;
- an in-range `original_variant_index`;
- at least the requested number of variants; and
- embedding coverage for every sequence ID required by the training and test
  partitions.

After preflight succeeds, the existing setup and fold-loader construction may
run. Reopening a read-only NumPy memmap and rereading a small CSV is acceptable;
avoiding partial run directories is more important than prematurely coupling
validation state to runtime loader objects.

### 4. Canonical-only baseline evaluation

The canonical embedding is the packed view identified by
`metadata.original_variant_index`. In the current public packed artifact that
index is `0`, but evaluation must read the metadata rather than hardcode it.

Baseline evaluation explicitly requests canonical-only test data, regardless
of `training.use_variants`. `SimpleDataset` then returns a two-dimensional
`[batch, embedding_dim]` tensor selected by `original_variant_index`.

Training remains unchanged:

- variant sampling keeps the canonical embedding first;
- BCE uses only that first view;
- InfoNCE uses all sampled views; and
- OOF probabilities and thresholds remain canonical-classification outputs.

Prototype evaluation retains its existing variant behavior. The shared test
loader therefore needs an explicit evaluation-time variant override rather
than inferring baseline behavior solely from `training.use_variants`.

Multi-view probability averaging is deliberately excluded. If added later, it
must be applied consistently to validation/OOF and test predictions so threshold
calibration uses the same score distribution.

## Interfaces and File Responsibilities

- `src/effector_bincls/data/contracts.py`
  - Owns global labeled CSV validation.
- `src/data/dataset_construction/combine_pos_and_neg_csv.py`
  - Owns deterministic unique-union materialization of the pretraining runtime
    CSV while retaining provenance memberships in the full snapshot.
- `src/effector_bincls/training/validation.py`
  - Owns complete Contrastive-BCE config and preflight input validation.
- `src/effector_bincls/training/contrastive_bce.py`
  - Invokes both preflight layers before `setup_training`.
- `src/effector_bincls/training/data.py`
  - Builds folds and loaders after validation and supports an explicit
    evaluation-time variant override.
- `src/effector_bincls/evaluation/baseline.py`
  - Requests canonical-only data.
- `src/effector_bincls/evaluation/prototype.py`
  - Continues to use the configured prototype variant behavior.

No new public CLI flags or artifact fields are required.

## Error Handling and Observability

Preflight failures raise explicit `ValueError`, `FileNotFoundError`, or metadata
contract errors before logging infrastructure or result directories exist.
Messages include the qualified config field or source path and the expected
contract. Large ID collections are summarized with a bounded preview and total
count.

Runtime training exceptions keep the existing structured logging and traceback
behavior. There is no warning-and-continue path for invalid data or metadata.

## Testing

### Global contract tests

Add unit coverage for null and empty IDs, duplicate IDs within one partition,
cross-partition duplicates, conflicting duplicate labels, null labels,
non-binary and non-integer labels, and null or empty partitions. Verify that
tracked supported runtime CSVs satisfy the tightened contract.

Add provenance tests that preserve the intentional `train`/`pretrain`
membership overlap in `effector_dataset.csv`, reject inconsistent repeated
records, verify the pretraining runtime CSV is unique by `sequence_id`, and
verify its unique ID/label mapping equals the union of the two provenance
memberships. Regenerate the tracked pretraining runtime CSV with the corrected
construction logic.

### Contrastive-BCE preflight tests

For every invalid config or input, invoke the public entrypoint and assert both
the actionable exception and absence of `data.results_dir`. Cover missing
required fields, invalid enums and ranges, missing packed files, pooling and
dimension mismatches, invalid original index, too few variants, missing
sequence embeddings, one-class training data, and class counts below the fold
count.

### Canonical evaluation tests

Verify that a nonzero `original_variant_index` produces the expected canonical
two-dimensional batch for baseline evaluation. Verify exact equality of logits
and probabilities between direct canonical input and the previous multi-view
path. Retain end-to-end smoke coverage for checkpoint loading, evaluation, and
`test_evaluation.yaml` generation.

### Regression suite

Run the unit, smoke, full pytest, Ruff format, and Ruff check commands defined
in `AGENTS.md`. GPU-specific validation is not required.

## Documentation

Update:

- `docs/VALIDATION_GUIDE.md` for the global labeled runtime CSV contract;
- `docs/BASELINE_README.md` for Contrastive-BCE preflight and canonical-only
  evaluation;
- `docs/PROTOTYPE_RANKING_README.md` to note that global labeled validation also
  applies to prototype workflows;
- `docs/DATASET_CONSTRUCTION_GUIDE.md` and
  `src/data/dataset_construction/README.md` for provenance membership versus
  unique runtime materialization; and
- `README.md` for the concise public workflow contract.

Documentation must state that invalid inputs fail before run creation and that
noncanonical views regularize training but do not affect baseline evaluation
probabilities.

## Risks and Mitigations

- **Compatibility:** Previously tolerated malformed CSVs will fail. This is an
  intentional contract correction; errors are actionable and there is no
  fallback mode.
- **Repository data:** Regenerating `effector_pretrain_dataset.csv` removes 547
  redundant rows but no unique IDs. Tests compare ID-to-label mappings and
  train/test relationships before accepting the regenerated artifact.
- **Provenance semantics:** The full construction snapshot retains overlapping
  membership rows and is validated with provenance-specific checks rather than
  weakened runtime-loader rules.
- **External row-based consumers:** Consumers that bypass the package loader
  will no longer see duplicate weighting for 547 positives. This is documented
  as a correction; the package loader already removes that multiplicity.
- **Shared loader behavior:** Tests cover baseline, prototype single-stage,
  prototype two-stage, evaluation, analysis, and smoke workflows to detect
  unintended global effects.
- **Evaluation shape:** Exact-logit regression tests prove that canonical-only
  evaluation preserves predictions while reducing work.
- **Validation duplication:** Preflight may reopen read-only inputs during
  loader construction. This small cost avoids stateful validation objects and
  partial artifacts.

## Acceptance Criteria

1. Invalid labeled runtime data is rejected globally without coercion or silent
   deduplication.
2. The provenance snapshot retains valid overlapping membership rows, while
   the pretraining runtime CSV contains the same unique union exactly once.
3. Regenerating the pretraining runtime CSV preserves every unique ID, label,
   finetuning-train membership, and test-set exclusion.
4. Every Contrastive-BCE config, CSV, or packed-embedding contract failure
   occurs before `results_dir` is created.
5. Infeasible stratified folds are rejected before training.
6. Contrastive-BCE baseline evaluation loads only the metadata-designated
   canonical view.
7. Canonical-only evaluation probabilities match the current canonical logits
   exactly.
8. Prototype variant evaluation behavior and all artifact formats remain
   compatible.
9. The full CPU test and Ruff suites pass.
