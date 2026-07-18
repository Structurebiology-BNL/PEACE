# Contrastive-BCE Contract and Evaluation Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PEACE runtime datasets unambiguous, materialize the pretraining superset with one row per sequence, fail Contrastive-BCE before creating run artifacts when inputs are invalid, and evaluate baseline checkpoints on canonical embeddings only.

**Architecture:** Put universal labeled-runtime validation in `effector_bincls.data.contracts`, while keeping repeated membership rows legal only in provenance construction data. Add pure construction and Contrastive-BCE preflight helpers so their behavior is unit-testable before orchestration. Add an explicit evaluation-time variant override to the shared test loader; baseline evaluation requests canonical embeddings and prototype evaluation retains its existing behavior.

**Tech Stack:** Python 3.12, pandas, NumPy, PyTorch, scikit-learn, `ml_collections.ConfigDict`, pytest, Ruff, uv.

## Global Constraints

- Work only in `/tmp/peace-pr1-contract-evaluation` on branch `fix/pr1-contract-evaluation`; do not modify the dirty `main` checkout in `/home/xdai/PEACE`.
- Follow TDD for every persistent behavior change: observe the intended test fail before changing production code.
- Do not add third-party dependencies or change `pyproject.toml`/`uv.lock`.
- `load_labeled_dataset` must enforce the runtime contract globally with no opt-out, silent coercion, row skipping, or implicit de-duplication.
- Runtime sequence IDs must be unique across the entire CSV; labels must be non-null numeric integers in `{0, 1}`; sequence IDs and partitions must be non-null and non-blank.
- `effector_dataset.csv` and `combined_positives.csv` remain provenance snapshots and may represent one sequence in both `train` and `pretrain`; read them directly with pandas rather than the runtime loader.
- `effector_pretrain_dataset.csv` must contain one deterministic row per sequence in the union of provenance `train` and `pretrain`, with every output partition relabeled to `train`.
- The pretraining runtime set must remain a superset of the fine-tuning training set, with identical labels for shared IDs.
- All Contrastive-BCE configuration and input checks must finish before `setup_training` creates a results directory.
- The canonical embedding is selected from packed metadata `original_variant_index`; never hardcode variant index `0`.
- Baseline evaluation is canonical-only. Prototype evaluation keeps its existing variant behavior. Multi-view probability averaging and uncertainty reporting are out of scope.
- CPU tests are required; no CUDA-specific test is required.
- Preserve public command names, checkpoint layouts, result artifact names, and the import `effector_bincls.training.contrastive_bce.validate_contrastive_bce_config`.
- Use `UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run ...` for project commands.

---

## File Structure

- `src/effector_bincls/data/contracts.py` — owns the strict global runtime CSV contract.
- `src/data/dataset_construction/combine_pos_and_neg_csv.py` — owns provenance membership validation and deterministic pretraining runtime materialization.
- `src/effector_bincls/training/validation.py` — owns complete Contrastive-BCE config and input preflight validation.
- `src/effector_bincls/training/contrastive_bce.py` — preserves the public validator import and orders both preflight phases before run setup.
- `src/effector_bincls/training/__init__.py` — exports both Contrastive-BCE validation entry points from their canonical module.
- `src/effector_bincls/training/data.py` — exposes an explicit evaluation-time `use_variants_override` without changing default behavior.
- `src/effector_bincls/evaluation/baseline.py` — requests canonical-only test embeddings.
- `src/data/csv_dataset/effector_pretrain_dataset.csv` — tracked unique runtime union artifact.
- `tests/unit/data/test_datasets.py` — strict runtime contract and loader-override regression tests.
- `tests/unit/data/test_dataset_construction.py` — pure materialization and conflict-validation tests.
- `tests/unit/data/test_contracts.py` — tracked artifact/provenance relationship tests.
- `tests/unit/training/test_entrypoints.py` — complete config, input-preflight ordering, and packed-artifact regression tests.
- `tests/unit/evaluation/test_evaluation_entrypoints.py` — baseline canonical-request regression test.
- `README.md`, `docs/BASELINE_README.md`, `docs/PROTOTYPE_RANKING_README.md`, `docs/VALIDATION_GUIDE.md`, `docs/DATASET_CONSTRUCTION_GUIDE.md`, and `src/data/dataset_construction/README.md` — public contract and workflow documentation.

### Task 1: Enforce the Global Labeled Runtime CSV Contract

**Files:**
- Modify: `tests/unit/data/test_datasets.py`
- Modify: `src/effector_bincls/data/contracts.py`

**Interfaces:**
- Consumes: `load_labeled_dataset(csv_path, *, label_config=None, required_partitions=None) -> pd.DataFrame`.
- Produces: the same signature and returned columns/order, with fail-fast validation of required values, global ID uniqueness, binary labels, and requested partitions.

- [ ] **Step 1: Add focused failing contract tests**

Append these cases to `tests/unit/data/test_datasets.py`:

```python
@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([",1,train"], "null or blank sequence IDs"),
        (["   ,1,train"], "null or blank sequence IDs"),
        (["seq0,,train"], "null labels"),
        (["seq0,positive,train"], "numeric integer labels"),
        (["seq0,0.5,train"], "numeric integer labels"),
        (["seq0,2,train"], "labels outside \{0, 1\}"),
        (["seq0,1,"], "null or blank partitions"),
        (["seq0,1,   "], "null or blank partitions"),
        (["seq0,1,train", "seq0,1,train"], "duplicate sequence IDs.*seq0"),
        (["seq0,1,train", "seq0,0,test"], "duplicate sequence IDs.*seq0"),
    ],
)
def test_load_labeled_dataset_rejects_invalid_runtime_values(
    tmp_path: Path,
    rows: list[str],
    message: str,
) -> None:
    csv_path = _write_dataset_csv(tmp_path / "dataset.csv", rows)

    with pytest.raises(ValueError, match=message):
        load_labeled_dataset(csv_path)


def test_load_labeled_dataset_accepts_unique_binary_runtime_rows(
    tmp_path: Path,
) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        ["seq0,1,train", "seq1,0,test"],
    )

    result = load_labeled_dataset(
        csv_path,
        required_partitions={"train", "test"},
    )

    assert result["sequence_id"].tolist() == ["seq0", "seq1"]
    assert result["label"].tolist() == [1, 0]
```

- [ ] **Step 2: Run the new tests and verify the current loader accepts invalid data**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_datasets.py -k 'load_labeled_dataset'
```

Expected: the existing missing-column/partition tests pass and the new invalid-value cases fail because the loader currently returns those frames.

- [ ] **Step 3: Add small validation helpers and call them before partition validation**

Add NumPy and pandas dtype imports in `src/effector_bincls/data/contracts.py`, then add these helpers above `load_labeled_dataset`:

```python
import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _sample_values(series: pd.Series, mask: pd.Series) -> list[object]:
    return series.loc[mask].head(5).tolist()


def _validate_non_blank_column(
    df: pd.DataFrame,
    column: str,
    *,
    description: str,
    path: Path,
) -> None:
    invalid = df[column].isna() | df[column].astype(str).str.strip().eq("")
    if invalid.any():
        rows = df.index[invalid].tolist()[:5]
        raise ValueError(
            f"Runtime dataset {path} contains null or blank {description} "
            f"at row indices {rows}."
        )


def _validate_binary_labels(df: pd.DataFrame, column: str, path: Path) -> None:
    labels = df[column]
    if labels.isna().any():
        rows = df.index[labels.isna()].tolist()[:5]
        raise ValueError(
            f"Runtime dataset {path} contains null labels at row indices {rows}."
        )
    if is_bool_dtype(labels.dtype) or not is_numeric_dtype(labels.dtype):
        values = labels.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"Runtime dataset {path} requires numeric integer labels in {{0, 1}}; "
            f"got dtype {labels.dtype} with sample values {values}."
        )

    numeric_labels = labels.to_numpy(dtype=float)
    invalid_numeric = ~np.isfinite(numeric_labels) | ~np.equal(
        numeric_labels, np.floor(numeric_labels)
    )
    if invalid_numeric.any():
        values = labels.loc[invalid_numeric].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"Runtime dataset {path} requires numeric integer labels in {{0, 1}}; "
            f"got sample values {values}."
        )

    invalid = ~labels.isin([0, 1])
    if invalid.any():
        values = sorted(set(_sample_values(labels, invalid)))
        raise ValueError(
            f"Runtime dataset {path} contains labels outside {{0, 1}}: {values}."
        )


def _validate_unique_sequence_ids(
    df: pd.DataFrame,
    sequence_id_column: str,
    path: Path,
) -> None:
    duplicated = df[sequence_id_column].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = (
            df.loc[duplicated, sequence_id_column].drop_duplicates().head(5).tolist()
        )
        raise ValueError(
            f"Runtime dataset {path} contains duplicate sequence IDs: "
            f"{duplicate_ids}."
        )
```

After required-column validation in `load_labeled_dataset`, call:

```python
    _validate_non_blank_column(
        df,
        sequence_id_column,
        description="sequence IDs",
        path=path,
    )
    _validate_binary_labels(df, label_column, path)
    _validate_non_blank_column(
        df,
        DEFAULT_PARTITION_COLUMN,
        description="partitions",
        path=path,
    )
    _validate_unique_sequence_ids(df, sequence_id_column, path)
```

Keep requested-partition validation after these global checks. Do not strip, fill, cast, drop, or de-duplicate returned values.

- [ ] **Step 4: Run the focused and two-stage contract tests**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_datasets.py tests/unit/training/test_validation.py
```

Expected: PASS. The tracked pretraining artifact test is intentionally deferred until Task 2 because strict uniqueness will expose its 547 repeated IDs.

- [ ] **Step 5: Commit the global runtime contract**

```bash
git add src/effector_bincls/data/contracts.py tests/unit/data/test_datasets.py
git commit -m "fix: enforce labeled runtime CSV contracts"
```

### Task 2: Materialize a Unique Pretraining Runtime Superset

**Files:**
- Create: `tests/unit/data/test_dataset_construction.py`
- Modify: `src/data/dataset_construction/combine_pos_and_neg_csv.py`
- Modify: `tests/unit/data/test_contracts.py`
- Modify: `src/data/csv_dataset/effector_pretrain_dataset.csv`

**Interfaces:**
- Consumes: the construction snapshot columns `sequence_id`, `sequence`, `label`, `partition`, plus any retained provenance columns such as `cluster_id`.
- Produces: `build_pretraining_runtime_dataset(combined_df: pd.DataFrame) -> pd.DataFrame`, preserving first-seen order and non-partition fields while returning one `train` row per ID in the `train`/`pretrain` union.

- [ ] **Step 1: Add failing pure construction tests**

Create `tests/unit/data/test_dataset_construction.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.dataset_construction.combine_pos_and_neg_csv import (
    build_pretraining_runtime_dataset,
)


def test_build_pretraining_runtime_dataset_materializes_unique_union() -> None:
    combined = pd.DataFrame(
        [
            ("pretrain-only", "AAA", 1, "pretrain", 10.0),
            ("shared", "BBB", 1, "pretrain", 11.0),
            ("shared", "BBB", 1, "train", 11.0),
            ("train-negative", "CCC", 0, "train", None),
            ("held-out", "DDD", 1, "test", 12.0),
        ],
        columns=["sequence_id", "sequence", "label", "partition", "cluster_id"],
    )

    result = build_pretraining_runtime_dataset(combined)

    expected = pd.DataFrame(
        [
            ("pretrain-only", "AAA", 1, "train", 10.0),
            ("shared", "BBB", 1, "train", 11.0),
            ("train-negative", "CCC", 0, "train", None),
        ],
        columns=combined.columns,
    )
    assert_frame_equal(result.reset_index(drop=True), expected)
    assert result["sequence_id"].is_unique


@pytest.mark.parametrize(
    ("changed_column", "changed_value"),
    [("sequence", "DIFFERENT"), ("label", 0), ("cluster_id", 99.0)],
)
def test_build_pretraining_runtime_dataset_rejects_conflicting_identity_rows(
    changed_column: str,
    changed_value: object,
) -> None:
    combined = pd.DataFrame(
        [
            ("shared", "BBB", 1, "pretrain", 11.0),
            ("shared", "BBB", 1, "train", 11.0),
        ],
        columns=["sequence_id", "sequence", "label", "partition", "cluster_id"],
    )
    combined.loc[1, changed_column] = changed_value

    with pytest.raises(ValueError, match=f"shared.*{changed_column}"):
        build_pretraining_runtime_dataset(combined)


def test_build_pretraining_runtime_dataset_rejects_duplicate_membership() -> None:
    combined = pd.DataFrame(
        [
            ("shared", "BBB", 1, "train"),
            ("shared", "BBB", 1, "train"),
        ],
        columns=["sequence_id", "sequence", "label", "partition"],
    )

    with pytest.raises(ValueError, match="shared.*train.*pretrain"):
        build_pretraining_runtime_dataset(combined)
```

- [ ] **Step 2: Run the new construction tests and verify the helper is absent**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_dataset_construction.py
```

Expected: test collection fails because `build_pretraining_runtime_dataset` does not exist.

- [ ] **Step 3: Implement deterministic provenance validation and materialization**

Add this pure helper to `src/data/dataset_construction/combine_pos_and_neg_csv.py`:

```python
def build_pretraining_runtime_dataset(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Materialize the unique train/pretrain union for runtime pretraining."""
    required_columns = {"sequence_id", "sequence", "label", "partition"}
    missing_columns = sorted(required_columns - set(combined_df.columns))
    if missing_columns:
        raise ValueError(
            f"Combined provenance dataset is missing columns: {missing_columns}."
        )

    duplicated = combined_df["sequence_id"].duplicated(keep=False)
    identity_columns = [
        column for column in combined_df.columns if column != "partition"
    ]
    for sequence_id, group in combined_df.loc[duplicated].groupby(
        "sequence_id", sort=False
    ):
        memberships = group["partition"].tolist()
        if len(group) != 2 or set(memberships) != {"train", "pretrain"}:
            raise ValueError(
                f"Repeated provenance sequence ID {sequence_id!r} must have exactly "
                "one train and one pretrain membership; "
                f"got {memberships}."
            )
        for column in identity_columns:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Repeated provenance sequence ID {sequence_id!r} has "
                    f"conflicting {column!r} values."
                )

    runtime_df = combined_df.loc[
        combined_df["partition"].isin(["train", "pretrain"])
    ].copy()
    runtime_df = runtime_df.drop_duplicates(subset=["sequence_id"], keep="first")
    runtime_df["partition"] = "train"
    return runtime_df
```

Replace the existing inline pretraining selection with:

```python
    pretrain_df = build_pretraining_runtime_dataset(combined_df)
    pretrain_df.to_csv(pretrain_output_path, index=False)
```

Do not call `load_labeled_dataset` on `combined_df`; it is intentionally a provenance representation.

- [ ] **Step 4: Run the pure tests and observe the tracked provenance test still fail**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_dataset_construction.py tests/unit/data/test_contracts.py
```

Expected: pure construction tests pass; `test_contracts.py` fails because it still treats provenance data as runtime data and the tracked pretraining CSV has not been regenerated.

- [ ] **Step 5: Separate provenance reads from runtime validation in tracked-artifact tests**

In `tests/unit/data/test_contracts.py`, add:

```python
def _read_provenance_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sequence_id": str})
```

Use `_read_provenance_csv` for `combined_positives.csv` and `effector_dataset.csv`. Continue using `load_labeled_dataset` for the pretraining and fine-tuning runtime CSVs. Replace the expected pretraining calculation with:

```python
    expected_pretrain = effector_dataset[
        effector_dataset["partition"].isin({"train", "pretrain"})
    ].copy()
    repeated = expected_pretrain[
        expected_pretrain["sequence_id"].duplicated(keep=False)
    ]
    identity_columns = [
        column for column in expected_pretrain.columns if column != "partition"
    ]
    for _, group in repeated.groupby("sequence_id"):
        assert set(group["partition"]) == {"train", "pretrain"}
        assert all(group[column].nunique(dropna=False) == 1 for column in identity_columns)

    expected_pretrain = expected_pretrain.drop_duplicates(
        subset=["sequence_id"], keep="first"
    )
    expected_pretrain["partition"] = "train"
    assert expected_pretrain["sequence_id"].is_unique
    assert set(effector_finetune.loc[
        effector_finetune["partition"] == "train", "sequence_id"
    ]).issubset(set(effector_pretrain["sequence_id"]))
    assert set(effector_pretrain["sequence_id"]).isdisjoint(
        set(effector_finetune.loc[
            effector_finetune["partition"] == "test", "sequence_id"
        ])
    )
```

Retain the existing `assert_frame_equal` after normalizing the new expected frame.

- [ ] **Step 6: Regenerate the tracked runtime CSV deterministically**

Run the documented construction command:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.data.dataset_construction.combine_pos_and_neg_csv \
  --positive-csv src/data/dataset_construction/combined_positives.csv \
  --negative-csv src/data/dataset_construction/filtered_new_negative_representatives.csv \
  --pretrain-csv src/data/csv_dataset/effector_pretrain_dataset.csv \
  --finetune-csv /tmp/peace-pr1-contract-evaluation-finetune-check.csv \
  --negative-ratio 50 \
  --random-seed 42
```

Compare the temporary fine-tuning output without overwriting the tracked file:

```bash
cmp src/data/csv_dataset/effector_finetune_dataset.csv /tmp/peace-pr1-contract-evaluation-finetune-check.csv
```

Expected: `cmp` exits 0. The tracked pretraining CSV drops exactly 547 redundant membership rows, going from 33,231 rows to 32,684 rows; all 32,684 sequence IDs are unique, labels remain 31,960 negative and 724 positive, and all partitions are `train`.

- [ ] **Step 7: Run all data contract tests**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data
```

Expected: PASS.

- [ ] **Step 8: Commit construction logic, artifact, and tests**

```bash
git add src/data/dataset_construction/combine_pos_and_neg_csv.py src/data/csv_dataset/effector_pretrain_dataset.csv tests/unit/data/test_dataset_construction.py tests/unit/data/test_contracts.py
git commit -m "fix: materialize unique pretraining runtime data"
```

### Task 3: Complete the Contrastive-BCE Configuration Contract

**Files:**
- Modify: `tests/unit/training/test_entrypoints.py`
- Modify: `src/effector_bincls/training/validation.py`
- Modify: `src/effector_bincls/training/contrastive_bce.py`
- Modify: `src/effector_bincls/training/__init__.py`

**Interfaces:**
- Consumes: the public `ConfigDict` schema in `src/configs/contrastive_bce.yaml`.
- Produces: `validate_contrastive_bce_config(config: ConfigDict) -> None` in `training.validation`, re-exported from both `training` and `training.contrastive_bce` for backward compatibility.

- [ ] **Step 1: Expand the valid test fixture and invalid-field matrix**

Extend `_valid_contrastive_bce_config` in `tests/unit/training/test_entrypoints.py` to mirror the full public YAML:

```python
            "training": {
                "batch_size": 2,
                "num_folds": 2,
                "threshold_method": "youden",
                "target_recall": 0.85,
                "num_epochs": 2,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "grad_clip_value": 2.0,
                "loss_type": "contrastive_bce",
                "bce_weight": 1.0,
                "unsupervised_weight": 0.5,
                "temperature": 0.07,
                "use_variants": True,
                "variant_sampling": {
                    "enabled": True,
                    "num_variants": 2,
                    "always_include_original": True,
                },
                "monitor_metric": "auprc",
                "mode": "max",
                "lr_scheduler": {
                    "scheduler_type": "cosine",
                    "eta_min": 1e-7,
                },
            },
            "output": {
                "save_checkpoints": False,
                "plot_training_curves": False,
            },
```

Add these entries to the existing invalid-value parameterization:

```python
        (("features", "normalize"), "false", "features.normalize must be boolean"),
        (("features", "pooling_type"), "", "features.pooling_type must be non-empty"),
        (("model", "dropout_rate"), 1.0, "model.dropout_rate must be in \[0, 1\)"),
        (("training", "learning_rate"), 0.0, "training.learning_rate must be finite and > 0"),
        (("training", "weight_decay"), -0.1, "training.weight_decay must be finite and >= 0"),
        (("training", "warmup_epochs"), 3, "training.warmup_epochs must not exceed training.num_epochs"),
        (("training", "early_stopping_patience"), 0, "training.early_stopping_patience must be a positive integer"),
        (("training", "grad_clip_value"), float("inf"), "training.grad_clip_value must be finite and > 0"),
        (("training", "threshold_method"), "accuracy", "training.threshold_method must be one of"),
        (("training", "target_recall"), 1.1, "training.target_recall must be in \(0, 1\]"),
        (("training", "monitor_metric"), "accuracy", "training.monitor_metric must be one of"),
        (("training", "mode"), "min", "training.mode='max'.*monitor_metric='auprc'"),
        (("training", "lr_scheduler", "scheduler_type"), "linear", "scheduler_type must be one of"),
        (("training", "lr_scheduler", "eta_min"), -1.0, "eta_min must be finite and >= 0"),
        (("output", "save_checkpoints"), 1, "output.save_checkpoints must be boolean"),
        (("hardware", "gpu_id"), -2, "hardware.gpu_id must be an integer >= -1"),
        (("hardware", "random_seed"), -1, "hardware.random_seed must be an integer >= 0"),
        (("hardware", "num_workers"), -1, "hardware.num_workers must be an integer >= 0"),
```

Add this reusable public-entrypoint assertion below `_set_nested`:

```python
def _assert_entrypoint_rejects_without_results(
    monkeypatch,
    tmp_path: Path,
    config: ConfigDict,
    message: str,
    *,
    results_dir: Path | None = None,
) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    expected_results_dir = results_dir or Path(config.data.results_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-contrastive-bce", "--config", str(config_path)],
    )

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        contrastive_bce_main()

    assert not expected_results_dir.exists()
```

Change the existing invalid-value test to use a unique temporary root and invoke the public entrypoint for every parameterized case:

```python
def test_contrastive_bce_entrypoint_rejects_invalid_config_before_setup(
    monkeypatch,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    _set_nested(raw_config, path, value)
    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        ConfigDict(raw_config),
        message,
    )
```

Also make missing-section validation go through the public entrypoint. Because deleting `data` also removes the results path, keep `data.results_dir` available in a separate variable and assert it is absent directly:

```python
@pytest.mark.parametrize("section", ["data", "features", "model", "training", "output", "hardware"])
def test_contrastive_bce_entrypoint_requires_public_sections_before_setup(
    monkeypatch,
    tmp_path: Path,
    section: str,
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    results_dir = Path(raw_config["data"]["results_dir"])
    del raw_config[section]
    config_path = tmp_path / f"missing-{section}.yml"
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train-contrastive-bce", "--config", str(config_path)],
    )

    with pytest.raises(ValueError, match=f"requires a {section} section"):
        contrastive_bce_main()

    assert not results_dir.exists()
```

Add `_delete_nested` and an exhaustive missing-required-field matrix so absent leaf values also fail through the public entrypoint:

```python
def _delete_nested(config: dict, path: tuple[str, ...]) -> None:
    section = config
    for key in path[:-1]:
        section = section[key]
    del section[path[-1]]


@pytest.mark.parametrize(
    "path",
    [
        ("data", "csv_path"),
        ("data", "embedding_dir"),
        ("data", "results_dir"),
        ("features", "normalize"),
        ("features", "pooling_type"),
        ("model", "type"),
        ("model", "input_dim"),
        ("model", "output_dim"),
        ("model", "dropout_rate"),
        ("model", "use_contrastive"),
        ("model", "contrastive_dim"),
        ("model", "encoder_hidden_dim"),
        ("training", "batch_size"),
        ("training", "num_folds"),
        ("training", "threshold_method"),
        ("training", "target_recall"),
        ("training", "num_epochs"),
        ("training", "learning_rate"),
        ("training", "weight_decay"),
        ("training", "warmup_epochs"),
        ("training", "early_stopping_patience"),
        ("training", "grad_clip_value"),
        ("training", "use_variants"),
        ("training", "variant_sampling"),
        ("training", "loss_type"),
        ("training", "bce_weight"),
        ("training", "unsupervised_weight"),
        ("training", "temperature"),
        ("training", "monitor_metric"),
        ("training", "mode"),
        ("training", "lr_scheduler"),
        ("output", "save_checkpoints"),
        ("output", "plot_training_curves"),
        ("hardware", "gpu_id"),
        ("hardware", "random_seed"),
        ("hardware", "deterministic"),
        ("hardware", "debug_logging"),
        ("hardware", "num_workers"),
    ],
)
def test_contrastive_bce_entrypoint_rejects_missing_fields_before_setup(
    monkeypatch,
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    raw_config = _valid_contrastive_bce_config(tmp_path).to_dict()
    results_dir = Path(raw_config["data"]["results_dir"])
    _delete_nested(raw_config, path)

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        ConfigDict(raw_config),
        path[-1],
        results_dir=results_dir,
    )
```

The nested `variant_sampling` and `lr_scheduler` leaf fields are covered by the invalid-value matrix (`None` is an invalid value); this matrix proves every top-level workflow field is required without duplicating the same branch.

- [ ] **Step 2: Run config validation tests and verify incomplete checks fail**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/training/test_entrypoints.py -k 'validate_contrastive_bce_config or invalid_config'
```

Expected: the new invalid-field and missing-section cases fail.

- [ ] **Step 3: Move and complete config validation in `training.validation`**

Add these typed helpers in `src/effector_bincls/training/validation.py`:

```python
import math
from typing import Any


def _require_section(container: Any, name: str, context: str) -> Any:
    section = getattr(container, name, None)
    if section is None:
        raise ValueError(f"{context} requires a {name} section.")
    return section


def _require_non_empty_string(section: Any, name: str, qualified_name: str) -> str:
    value = getattr(section, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{qualified_name} must be a non-empty string.")
    return value


def _require_bool(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    expected: bool | None = None,
) -> bool:
    value = getattr(section, name, None)
    if not isinstance(value, bool):
        raise ValueError(f"{qualified_name} must be boolean, got {value!r}.")
    if expected is not None and value is not expected:
        expected_text = str(expected).lower()
        raise ValueError(
            f"Contrastive-BCE requires {qualified_name}={expected_text}."
        )
    return value


def _integer_requirement(qualified_name: str, minimum: int) -> str:
    if minimum == 1:
        return f"{qualified_name} must be a positive integer"
    if minimum == 2:
        return f"{qualified_name} must be at least 2"
    return f"{qualified_name} must be an integer >= {minimum}"


def _require_integer(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    minimum: int,
) -> int:
    value = getattr(section, name, None)
    requirement = _integer_requirement(qualified_name, minimum)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{requirement}, got {value!r}.")
    return value


def _require_exact_integer(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    expected: int,
) -> int:
    value = getattr(section, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"Contrastive-BCE requires {qualified_name}={expected}.")
    return value


def _finite_requirement(
    qualified_name: str,
    minimum: float | None,
    maximum: float | None,
    minimum_inclusive: bool,
    maximum_inclusive: bool,
) -> str:
    if minimum == 0.0 and maximum is None:
        operator = ">=" if minimum_inclusive else ">"
        return f"{qualified_name} must be finite and {operator} 0"
    if minimum == 0.0 and maximum == 1.0:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        return f"{qualified_name} must be in {left}0, 1{right}"
    return f"{qualified_name} is outside its supported finite range"


def _require_finite(
    section: Any,
    name: str,
    qualified_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    value = getattr(section, name, None)
    requirement = _finite_requirement(
        qualified_name,
        minimum,
        maximum,
        minimum_inclusive,
        maximum_inclusive,
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{requirement}, got {value!r}.")
    numeric_value = float(value)
    invalid = not math.isfinite(numeric_value)
    if minimum is not None:
        invalid = invalid or (
            numeric_value < minimum
            if minimum_inclusive
            else numeric_value <= minimum
        )
    if maximum is not None:
        invalid = invalid or (
            numeric_value > maximum
            if maximum_inclusive
            else numeric_value >= maximum
        )
    if invalid:
        raise ValueError(f"{requirement}, got {value!r}.")
    return numeric_value


def _require_choice(
    section: Any,
    name: str,
    qualified_name: str,
    choices: set[str],
) -> str:
    value = getattr(section, name, None)
    if value not in choices:
        raise ValueError(
            f"{qualified_name} must be one of {sorted(choices)}, got {value!r}."
        )
    return value


def _validate_plateau_scheduler(scheduler: Any) -> None:
    if hasattr(scheduler, "factor"):
        _require_finite(
            scheduler,
            "factor",
            "training.lr_scheduler.factor",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
    for name in ("patience", "cooldown"):
        if hasattr(scheduler, name):
            _require_integer(
                scheduler,
                name,
                f"training.lr_scheduler.{name}",
                minimum=0,
            )
    for name in ("threshold", "min_lr"):
        if hasattr(scheduler, name):
            _require_finite(
                scheduler,
                name,
                f"training.lr_scheduler.{name}",
                minimum=0.0,
            )
    if hasattr(scheduler, "threshold_mode"):
        _require_choice(
            scheduler,
            "threshold_mode",
            "training.lr_scheduler.threshold_mode",
            {"rel", "abs"},
        )
```

Then implement `validate_contrastive_bce_config` with these exact rules:

```python
def validate_contrastive_bce_config(config: ConfigDict) -> None:
    """Validate the complete public Contrastive-BCE configuration contract."""
    data = _require_section(config, "data", "Contrastive-BCE")
    features = _require_section(config, "features", "Contrastive-BCE")
    model = _require_section(config, "model", "Contrastive-BCE")
    training = _require_section(config, "training", "Contrastive-BCE")
    output = _require_section(config, "output", "Contrastive-BCE")
    hardware = _require_section(config, "hardware", "Contrastive-BCE")

    validate_baseline_training_config(config)
    for name in ("csv_path", "embedding_dir", "results_dir"):
        _require_non_empty_string(data, name, f"data.{name}")
    _require_bool(features, "normalize", "features.normalize")
    _require_non_empty_string(features, "pooling_type", "features.pooling_type")

    _require_exact_integer(model, "output_dim", "model.output_dim", expected=1)
    _require_bool(model, "use_contrastive", "model.use_contrastive", expected=True)
    for name in ("input_dim", "encoder_hidden_dim", "contrastive_dim"):
        _require_integer(model, name, f"model.{name}", minimum=1)
    _require_finite(model, "dropout_rate", "model.dropout_rate", minimum=0.0, maximum=1.0, maximum_inclusive=False)

    _require_integer(training, "batch_size", "training.batch_size", minimum=1)
    _require_integer(training, "num_folds", "training.num_folds", minimum=2)
    num_epochs = _require_integer(training, "num_epochs", "training.num_epochs", minimum=1)
    _require_finite(training, "learning_rate", "training.learning_rate", minimum=0.0, minimum_inclusive=False)
    _require_finite(training, "weight_decay", "training.weight_decay", minimum=0.0)
    warmup_epochs = _require_integer(training, "warmup_epochs", "training.warmup_epochs", minimum=0)
    if warmup_epochs > num_epochs:
        raise ValueError("training.warmup_epochs must not exceed training.num_epochs.")
    _require_integer(training, "early_stopping_patience", "training.early_stopping_patience", minimum=1)
    _require_finite(training, "grad_clip_value", "training.grad_clip_value", minimum=0.0, minimum_inclusive=False)
    _require_choice(training, "threshold_method", "training.threshold_method", {"youden", "f1", "mcc", "recall_constrained"})
    _require_finite(training, "target_recall", "training.target_recall", minimum=0.0, maximum=1.0, minimum_inclusive=False)
    _require_choice(training, "monitor_metric", "training.monitor_metric", {"loss", "auprc", "roc_auc"})
    mode = _require_choice(training, "mode", "training.mode", {"min", "max"})
    expected_mode = "min" if training.monitor_metric == "loss" else "max"
    if mode != expected_mode:
        raise ValueError(
            f"Contrastive-BCE requires training.mode='{expected_mode}' for "
            f"monitor_metric='{training.monitor_metric}'."
        )

    if getattr(training, "loss_type", None) != "contrastive_bce":
        raise ValueError("Contrastive-BCE requires training.loss_type='contrastive_bce'.")
    for name in ("bce_weight", "unsupervised_weight", "temperature"):
        _require_finite(training, name, f"training.{name}", minimum=0.0, minimum_inclusive=False)
    _require_bool(training, "use_variants", "training.use_variants", expected=True)
    variant_sampling = _require_section(training, "variant_sampling", "Contrastive-BCE training")
    _require_bool(variant_sampling, "enabled", "training.variant_sampling.enabled", expected=True)
    _require_integer(variant_sampling, "num_variants", "training.variant_sampling.num_variants", minimum=2)
    _require_bool(variant_sampling, "always_include_original", "training.variant_sampling.always_include_original", expected=True)

    scheduler = _require_section(training, "lr_scheduler", "Contrastive-BCE training")
    scheduler_type = _require_choice(scheduler, "scheduler_type", "training.lr_scheduler.scheduler_type", {"plateau", "cosine"})
    if scheduler_type == "cosine":
        eta_min = _require_finite(scheduler, "eta_min", "training.lr_scheduler.eta_min", minimum=0.0)
        if eta_min >= float(training.learning_rate):
            raise ValueError("training.lr_scheduler.eta_min must be less than training.learning_rate.")
    else:
        _validate_plateau_scheduler(scheduler)

    _require_bool(output, "save_checkpoints", "output.save_checkpoints")
    _require_bool(output, "plot_training_curves", "output.plot_training_curves")
    _require_integer(hardware, "gpu_id", "hardware.gpu_id", minimum=-1)
    _require_integer(hardware, "random_seed", "hardware.random_seed", minimum=0)
    _require_bool(hardware, "deterministic", "hardware.deterministic")
    _require_bool(hardware, "debug_logging", "hardware.debug_logging")
    _require_integer(hardware, "num_workers", "hardware.num_workers", minimum=0)
```

The helpers reject booleans as integers, reject non-finite numbers, name the qualified field in every error, and return the validated value. The `plateau` branch validates any explicitly supplied runtime fields without requiring optional fields that the runtime already defaults.

Remove the old helper implementations from `training/contrastive_bce.py` and import the canonical validator there. Update `training/__init__.py` to import it from `training.validation`. This keeps the old module-level import working because the imported name remains bound in `contrastive_bce.py`.

- [ ] **Step 4: Run config tests and public-config tests**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/training/test_entrypoints.py tests/unit/training/test_validation.py tests/unit/test_public_config_surface.py
```

Expected: PASS, including the unchanged public `contrastive_bce.yaml`.

- [ ] **Step 5: Commit complete configuration validation**

```bash
git add src/effector_bincls/training/validation.py src/effector_bincls/training/contrastive_bce.py src/effector_bincls/training/__init__.py tests/unit/training/test_entrypoints.py
git commit -m "fix: validate the full contrastive BCE config"
```

### Task 4: Preflight Contrastive-BCE Data and Packed Embeddings Before Run Setup

**Files:**
- Modify: `tests/unit/training/test_entrypoints.py`
- Modify: `src/effector_bincls/training/validation.py`
- Modify: `src/effector_bincls/training/contrastive_bce.py`
- Modify: `src/effector_bincls/training/__init__.py`

**Interfaces:**
- Consumes: `load_labeled_dataset`, `open_packed_embedding_dataset`, `require_sequence_indices`, and the validated config from Task 3.
- Produces: `validate_contrastive_bce_inputs(config: ConfigDict) -> None`, called after config validation and before `setup_training`.

- [ ] **Step 1: Add failing input-contract and ordering tests**

Import `validate_contrastive_bce_inputs` in `tests/unit/training/test_entrypoints.py` and add:

```python
def test_validate_contrastive_bce_inputs_accepts_complete_toy_data(
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)

    validate_contrastive_bce_inputs(config)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("duplicate", "duplicate sequence IDs.*seq0"),
        ("nonbinary", "labels outside \{0, 1\}"),
    ],
)
def test_contrastive_bce_entrypoint_rejects_invalid_runtime_csv_before_setup(
    monkeypatch,
    tmp_path: Path,
    invalid_kind: str,
    message: str,
) -> None:
    config = _write_toy_dataset(tmp_path)
    dataframe = pd.read_csv(config.data.csv_path)
    if invalid_kind == "duplicate":
        dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)
    else:
        dataframe.loc[0, "label"] = 2
    dataframe.to_csv(config.data.csv_path, index=False)

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        message,
    )


@pytest.mark.parametrize(
    ("config_path", "value", "message"),
    [
        (("features", "pooling_type"), "max", "pooling_type.*mean.*max"),
        (("model", "input_dim"), 4, "embedding_dim.*3.*4"),
        (("training", "variant_sampling", "num_variants"), 3, "contains 2 variants.*requests 3"),
    ],
)
def test_validate_contrastive_bce_inputs_rejects_packed_contract_mismatch(
    monkeypatch,
    tmp_path: Path,
    config_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    raw_config = _write_toy_dataset(tmp_path).to_dict()
    _set_nested(raw_config, config_path, value)
    config = ConfigDict(raw_config)

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        message,
    )


def test_validate_contrastive_bce_inputs_rejects_invalid_canonical_index(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)
    metadata_path = Path(config.data.embedding_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["original_variant_index"] = 2
    metadata_path.write_text(json.dumps(metadata))

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        "original_variant_index.*2",
    )


def test_validate_contrastive_bce_inputs_requires_every_runtime_embedding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)
    sequence_ids_path = Path(config.data.embedding_dir) / "sequence_ids.txt"
    sequence_ids_path.write_text("seq0\nseq1\nseq2\nseq3\nseq4\nmissing-id\n")

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        "seq5",
    )


def test_validate_contrastive_bce_inputs_rejects_missing_packed_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)
    (Path(config.data.embedding_dir) / "metadata.json").unlink()

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        "metadata.json",
    )


def test_validate_contrastive_bce_inputs_requires_both_training_classes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)
    dataframe = pd.read_csv(config.data.csv_path)
    dataframe.loc[dataframe["partition"] == "train", "label"] = 1
    dataframe.to_csv(config.data.csv_path, index=False)

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        "train partition.*both labels",
    )


def test_invalid_training_class_counts_do_not_create_results_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_toy_dataset(tmp_path)
    config.training.num_folds = 3

    _assert_entrypoint_rejects_without_results(
        monkeypatch,
        tmp_path,
        config,
        "at least 3 training samples per class",
    )
```

Add `json` and `pandas as pd` imports. Keep `_write_toy_dataset` at four training rows (two per class) and two test rows (one per class).

- [ ] **Step 2: Run input tests and verify failures occur late or not at all**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/training/test_entrypoints.py -k 'contrastive_bce_inputs or invalid_training_class_counts'
```

Expected: the new validator import/test cases fail; for input errors currently detected by loader construction, the existing main path creates the results directory before raising.

- [ ] **Step 3: Implement packed-artifact and split preflight**

Add this function to `src/effector_bincls/training/validation.py`:

```python
def validate_contrastive_bce_inputs(config: ConfigDict) -> None:
    """Validate runtime data and packed embeddings without creating run output."""
    label_config = getattr(config.data, "label_config", {})
    sequence_id_column, label_column = resolve_label_columns(label_config)
    dataframe = load_labeled_dataset(
        config.data.csv_path,
        label_config=label_config,
        required_partitions={"train", "test"},
    )

    required_rows = dataframe[
        dataframe[DEFAULT_PARTITION_COLUMN].isin({"train", "test"})
    ]
    train_rows = required_rows[
        required_rows[DEFAULT_PARTITION_COLUMN] == "train"
    ]
    train_labels = set(train_rows[label_column].tolist())
    if train_labels != {0, 1}:
        raise ValueError(
            "Contrastive-BCE train partition must contain both labels 0 and 1; "
            f"got {sorted(train_labels)}."
        )

    class_counts = train_rows[label_column].value_counts()
    num_folds = int(config.training.num_folds)
    underfilled = {
        int(label): int(class_counts.get(label, 0))
        for label in (0, 1)
        if int(class_counts.get(label, 0)) < num_folds
    }
    if underfilled:
        raise ValueError(
            f"Contrastive-BCE requires at least {num_folds} training samples per "
            f"class for {num_folds}-fold stratification; got {underfilled}."
        )

    embeddings, available_ids, metadata = open_packed_embedding_dataset(
        config.data.embedding_dir
    )
    available_variants = int(embeddings.shape[1])
    embedding_dim = int(embeddings.shape[2])
    requested_variants = int(config.training.variant_sampling.num_variants)
    if metadata.get("pooling_type") != config.features.pooling_type:
        raise ValueError(
            "Packed embedding pooling_type "
            f"{metadata.get('pooling_type')!r} does not match configured "
            f"pooling_type {config.features.pooling_type!r}."
        )
    if embedding_dim != int(config.model.input_dim):
        raise ValueError(
            f"Packed embedding_dim {embedding_dim} does not match "
            f"model.input_dim {config.model.input_dim}."
        )
    if available_variants < requested_variants:
        raise ValueError(
            f"Packed embedding dataset contains {available_variants} variants but "
            f"training.variant_sampling requests {requested_variants}."
        )

    original_variant_index = metadata.get("original_variant_index")
    if (
        isinstance(original_variant_index, bool)
        or not isinstance(original_variant_index, int)
        or not 0 <= original_variant_index < available_variants
    ):
        raise ValueError(
            "Packed embedding original_variant_index must be an integer within "
            f"[0, {available_variants}); got {original_variant_index!r}."
        )

    requested_ids = required_rows[sequence_id_column].tolist()
    available_id_set = set(available_ids)
    missing_ids = [
        sequence_id
        for sequence_id in requested_ids
        if sequence_id not in available_id_set
    ]
    if missing_ids:
        raise FileNotFoundError(
            "Packed embedding dataset is missing runtime sequence IDs: "
            f"{missing_ids[:5]} ({len(missing_ids)} total)."
        )
    require_sequence_indices(requested_ids, available_ids)
    del embeddings
```

Import `DEFAULT_PARTITION_COLUMN`, `open_packed_embedding_dataset`, `require_sequence_indices`, and `resolve_label_columns` from `effector_bincls.data`. Keep the defensive checks in `create_contrastive_bce_data_loader_fn`; preflight does not replace lower-level validation.

- [ ] **Step 4: Invoke both validators before any run directory is created**

In `src/effector_bincls/training/contrastive_bce.py`, order `main` as:

```python
    preflight_config = ConfigDict(load_config(args.config))
    validate_contrastive_bce_config(preflight_config)
    validate_contrastive_bce_inputs(preflight_config)

    start_time = time.time()
    config, run_dir, logger = setup_training(config_path=args.config)
```

Import both functions from `training.validation`, and export `validate_contrastive_bce_inputs` from `training/__init__.py`.

- [ ] **Step 5: Run entrypoint, loader, and smoke tests**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/training/test_entrypoints.py tests/unit/training/test_contrastive_bce.py tests/smoke -k 'contrastive or entrypoint'
```

Expected: PASS. Invalid config, invalid class support, metadata mismatch, insufficient variants, invalid canonical index, and missing embeddings all fail before a results directory exists.

- [ ] **Step 6: Commit input preflight and orchestration ordering**

```bash
git add src/effector_bincls/training/validation.py src/effector_bincls/training/contrastive_bce.py src/effector_bincls/training/__init__.py tests/unit/training/test_entrypoints.py
git commit -m "fix: preflight contrastive BCE inputs"
```

### Task 5: Make Baseline Evaluation Canonical-Only

**Files:**
- Modify: `tests/unit/data/test_datasets.py`
- Modify: `tests/unit/evaluation/test_evaluation_entrypoints.py`
- Modify: `src/effector_bincls/training/data.py`
- Modify: `src/effector_bincls/evaluation/baseline.py`

**Interfaces:**
- Consumes: `load_test_data(config, logger=None, test_csv_path=None)` and packed `original_variant_index` metadata.
- Produces: `load_test_data(config, logger=None, test_csv_path=None, *, use_variants_override: bool | None = None) -> DataLoader`; `None` preserves current behavior, `False` selects only the canonical vector.

- [ ] **Step 1: Add a failing loader-override test**

Append to `tests/unit/data/test_datasets.py`:

```python
def test_load_test_data_can_override_variant_config_for_canonical_evaluation(
    tmp_path: Path,
) -> None:
    csv_path = _write_dataset_csv(
        tmp_path / "dataset.csv",
        ["seq0,1,train", "seq1,0,test"],
    )
    embedding_dir = _write_packed_embeddings(
        tmp_path / "embeddings",
        ["seq0", "seq1"],
        np.asarray(
            [
                [[100.0, 100.0], [1.0, 1.0]],
                [[200.0, 200.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        original_variant_index=1,
    )
    config = ConfigDict(
        {
            "data": {"csv_path": str(csv_path), "embedding_dir": str(embedding_dir)},
            "features": {"normalize": False, "pooling_type": "mean"},
            "model": {"type": "simple_predictor"},
            "training": {
                "batch_size": 2,
                "use_variants": True,
                "loss_type": "contrastive_bce",
                "variant_sampling": {
                    "enabled": True,
                    "num_variants": 2,
                    "always_include_original": True,
                },
            },
            "hardware": {"num_workers": 0, "random_seed": 42},
        }
    )

    features, labels = next(
        iter(load_test_data(config, use_variants_override=False))
    )

    assert features.shape == (1, 2)
    assert torch.equal(features, torch.tensor([[2.0, 2.0]]))
    assert torch.equal(labels, torch.tensor([[0.0]]))
```

- [ ] **Step 2: Add a failing baseline-entrypoint contract test**

Replace the function-only import in `tests/unit/evaluation/test_evaluation_entrypoints.py` with a module import and add:

```python
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from ml_collections import ConfigDict

from effector_bincls.evaluation import baseline as baseline_evaluation


class StopAfterDataLoad(RuntimeError):
    pass


def test_baseline_evaluation_requests_canonical_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yml").write_text("model: {}\n")
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("sequence_id,label,partition\nseq0,1,test\n")
    args = SimpleNamespace(
        run_dir=tmp_path,
        test_csv=test_csv,
        threshold_method="youden",
        target_recall=0.85,
    )
    config = ConfigDict(
        {
            "model": {"type": "simple_predictor"},
            "training": {"use_variants": True},
        }
    )

    monkeypatch.setattr(baseline_evaluation, "parse_evaluation_args", lambda _: args)
    monkeypatch.setattr(baseline_evaluation, "load_run_config", lambda _: config)
    monkeypatch.setattr(
        baseline_evaluation,
        "setup_logger",
        lambda **_: logging.getLogger("baseline-canonical-test"),
    )
    monkeypatch.setattr(baseline_evaluation, "resolve_device", lambda _: "cpu")

    def stop_after_load(*args, **kwargs):
        assert kwargs["use_variants_override"] is False
        raise StopAfterDataLoad

    monkeypatch.setattr(baseline_evaluation, "load_test_data", stop_after_load)

    with pytest.raises(StopAfterDataLoad):
        baseline_evaluation.main()
```

Retain the callable export assertions using `baseline_evaluation.main` and the existing prototype import.

In the same file, import `torch` and `SimplePredictor`, then add the exact prediction-equivalence regression:

```python
def test_canonical_only_logits_match_canonical_first_multiview_logits() -> None:
    torch.manual_seed(7)
    model = SimplePredictor(
        input_dim=3,
        output_dim=1,
        dropout_rate=0.0,
        use_contrastive=True,
        contrastive_dim=2,
        encoder_hidden_dim=4,
    )
    model.eval()
    canonical = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    noncanonical = torch.tensor(
        [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
        dtype=torch.float32,
    )
    multiview = torch.stack([canonical, noncanonical], dim=1)

    with torch.no_grad():
        canonical_logits, _ = model(canonical)
        multiview_logits, _ = model(multiview)

    assert torch.equal(canonical_logits, multiview_logits)
    assert torch.equal(
        torch.sigmoid(canonical_logits),
        torch.sigmoid(multiview_logits),
    )
```

- [ ] **Step 3: Run the focused tests and verify the override does not exist**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_datasets.py::test_load_test_data_can_override_variant_config_for_canonical_evaluation tests/unit/evaluation/test_evaluation_entrypoints.py::test_baseline_evaluation_requests_canonical_embeddings tests/unit/evaluation/test_evaluation_entrypoints.py::test_canonical_only_logits_match_canonical_first_multiview_logits
```

Expected: the override tests fail because `use_variants_override` is not accepted or passed; the exact-logit regression passes and records the current canonical-first semantics before the loader change.

- [ ] **Step 4: Add the explicit shared-loader override**

Change `load_test_data` in `src/effector_bincls/training/data.py` to:

```python
def load_test_data(
    config: ConfigDict,
    logger: logging.Logger | None = None,
    test_csv_path: Optional[Path] = None,
    *,
    use_variants_override: bool | None = None,
) -> DataLoader:
    """Load test data for package-native evaluation and analysis."""
```

Resolve variants with:

```python
    use_variants = (
        getattr(config.training, "use_variants", False)
        if use_variants_override is None
        else use_variants_override
    )
```

Leave dataset construction and contrastive collate selection unchanged. `False` causes `SimpleDataset` to select packed metadata `original_variant_index`, including when that index is not zero.

- [ ] **Step 5: Make only baseline evaluation request the override**

Change the call in `src/effector_bincls/evaluation/baseline.py` to:

```python
        test_loader = load_test_data(
            config,
            logger=logger,
            test_csv_path=args.test_csv,
            use_variants_override=False,
        )
```

Do not change the call in `src/effector_bincls/evaluation/prototype.py`.

- [ ] **Step 6: Run evaluation and data regression tests**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data/test_datasets.py tests/unit/evaluation
```

Expected: PASS. The existing test that returns multiple views by default remains green, proving prototype/default behavior was not globally changed.

- [ ] **Step 7: Commit canonical baseline evaluation**

```bash
git add src/effector_bincls/training/data.py src/effector_bincls/evaluation/baseline.py tests/unit/data/test_datasets.py tests/unit/evaluation/test_evaluation_entrypoints.py
git commit -m "fix: evaluate baselines on canonical embeddings"
```

### Task 6: Document the Contracts and Run Complete Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/BASELINE_README.md`
- Modify: `docs/PROTOTYPE_RANKING_README.md`
- Modify: `docs/VALIDATION_GUIDE.md`
- Modify: `docs/DATASET_CONSTRUCTION_GUIDE.md`
- Modify: `src/data/dataset_construction/README.md`

**Interfaces:**
- Consumes: behavior implemented in Tasks 1–5 and the approved design spec at `docs/superpowers/specs/2026-07-18-contrastive-bce-contract-evaluation-design.md`.
- Produces: public documentation that distinguishes runtime uniqueness from provenance membership, enumerates preflight failures, and states the evaluation view rule.

- [ ] **Step 1: Document runtime uniqueness and provenance membership**

Add this contract text to the runtime dataset sections of `README.md`, `docs/VALIDATION_GUIDE.md`, and `docs/DATASET_CONSTRUCTION_GUIDE.md`:

```markdown
Runtime labeled CSVs require non-empty unique sequence IDs, binary integer labels
(`0` or `1`), and non-empty partitions. Duplicate IDs are rejected globally,
including duplicates assigned to different partitions.

The construction snapshot `effector_dataset.csv` is different: repeated
`train`/`pretrain` rows encode membership provenance and are not a runtime input.
`effector_pretrain_dataset.csv` materializes the unique union of those memberships,
relabels every row to `train`, and remains a superset of the fine-tuning training
set. Shared provenance memberships must agree on sequence, label, and retained
metadata or construction fails.
```

Update `src/data/dataset_construction/README.md` so step 5 says the pretraining output has one row per sequence in the `train`/`pretrain` union. Preserve the existing regeneration command.

- [ ] **Step 2: Document preflight timing and validation coverage**

Add this paragraph to the Contrastive-BCE section of `docs/BASELINE_README.md`:

```markdown
Before a run directory is created, `train-contrastive-bce` validates the complete
configuration, runtime CSV contract, required train/test partitions, both training
classes, per-class support
for stratified folds, packed pooling and dimensionality, requested variant count,
canonical variant metadata, and embedding coverage for every train/test ID. Invalid
or ambiguous input fails without leaving a partial run directory.
```

In `docs/PROTOTYPE_RANKING_README.md`, state that prototype workflows use the same strict global labeled-runtime CSV contract, while the complete packed-artifact preflight described above belongs to the Contrastive-BCE entrypoint and does not change prototype single-stage/two-stage view semantics.

- [ ] **Step 3: Document canonical-only baseline evaluation**

Add this text to `docs/VALIDATION_GUIDE.md` and the evaluation section of `docs/BASELINE_README.md`:

```markdown
Baseline evaluation always uses one canonical embedding per sequence, selected by
the packed dataset's `original_variant_index`. A saved training config may enable
variants for Contrastive-BCE regularization, but baseline test probabilities are
not averaged across variants. Prototype evaluation retains its configured variant
behavior.
```

Do not describe multi-view averaging or uncertainty artifacts because neither is implemented.

- [ ] **Step 4: Format and lint before the documentation commit**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run ruff format
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run ruff check
```

Expected: Ruff reports all checks passed. Review any formatter changes and include only files belonging to this plan.

- [ ] **Step 5: Commit documentation and any mechanical formatting**

```bash
git add README.md docs/BASELINE_README.md docs/PROTOTYPE_RANKING_README.md docs/VALIDATION_GUIDE.md docs/DATASET_CONSTRUCTION_GUIDE.md src/data/dataset_construction/README.md src/effector_bincls/data/contracts.py src/data/dataset_construction/combine_pos_and_neg_csv.py src/effector_bincls/training/validation.py src/effector_bincls/training/contrastive_bce.py src/effector_bincls/training/__init__.py src/effector_bincls/training/data.py src/effector_bincls/evaluation/baseline.py tests/unit/data/test_datasets.py tests/unit/data/test_dataset_construction.py tests/unit/data/test_contracts.py tests/unit/training/test_entrypoints.py tests/unit/evaluation/test_evaluation_entrypoints.py
git commit -m "docs: clarify runtime and evaluation contracts"
```

- [ ] **Step 6: Audit the regenerated datasets directly**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import pandas as pd; p=pd.read_csv('src/data/csv_dataset/effector_pretrain_dataset.csv', dtype={'sequence_id': str}); f=pd.read_csv('src/data/csv_dataset/effector_finetune_dataset.csv', dtype={'sequence_id': str}); pids=set(p.sequence_id); ft=set(f.loc[f.partition.eq('train'),'sequence_id']); test=set(f.loc[f.partition.eq('test'),'sequence_id']); assert len(p)==32684; assert p.sequence_id.nunique()==32684; assert p.label.value_counts().to_dict()=={0:31960,1:724}; assert set(p.partition)=={'train'}; assert len(ft)==27897; assert len(pids-ft)==4787; assert len(test)==7038; assert ft<=pids; assert pids.isdisjoint(test); print('pretraining runtime contract: PASS')"
```

Expected: `pretraining runtime contract: PASS`.

- [ ] **Step 7: Run targeted, unit, smoke, and full regression suites**

Run in order:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit/data tests/unit/training/test_entrypoints.py tests/unit/training/test_validation.py tests/unit/evaluation
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/unit
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/smoke
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: every command passes. Record exact test counts and warnings in the handoff; do not claim success from an earlier baseline run.

- [ ] **Step 8: Verify public CLI compatibility**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run effector-bincls train-contrastive-bce --help
UV_PROJECT_ENVIRONMENT=/tmp/peace-pr1-contract-evaluation-venv UV_CACHE_DIR=/tmp/uv-cache uv run effector-bincls evaluate-baseline --help
```

Expected: both commands exit 0 and retain their existing public flags.

- [ ] **Step 9: Inspect the final diff and commit graph**

Run:

```bash
git status --short
git diff --check
git log --oneline --decorate -8
git diff --stat d40a353..HEAD
```

Expected: the worktree is clean; `git diff --check` emits no output; the six implementation commits follow the design and plan commits; no dependency, checkpoint, result, or unrelated files appear.

## Rollout Risks and Mitigations

- Strict global validation may reject external CSVs that previously ran with duplicates, blank values, or non-binary labels. This is intentional; error messages include the path and sample rows/IDs so callers can repair inputs explicitly.
- Unique materialization removes row-weighting caused by the 547 duplicated positive memberships. PEACE's two-stage loader already converted IDs through dictionaries/sets, so effective package membership and class counts are unchanged; the tracked artifact now makes that contract explicit for every consumer.
- Complete preflight reads the labeled CSV and opens packed metadata twice (preflight and loader creation). Both operations are bounded metadata/index work; do not scan all embedding values or add a cache until profiling shows a material cost.
- Canonical-only baseline evaluation changes the input shape and removes unnecessary noncanonical computation for Contrastive-BCE saved configs. The exact-logit regression guards the approved requirement that predictions remain identical to the current canonical-first classifier output.
- Repeated provenance rows are accepted only as an exact `train`/`pretrain` pair with identical non-partition fields. Any future construction design requiring other membership multiplicity must update the provenance contract explicitly rather than weakening runtime validation.

## Open Questions During Execution

- None requiring design approval. If a retained provenance duplicate violates the exact `train`/`pretrain` identity rule, stop and report the concrete IDs and differing columns rather than relaxing the rule.
- If a documented public config relies on a scheduler field outside the domains listed in Task 3, report the exact config and runtime consumer before broadening validation.
