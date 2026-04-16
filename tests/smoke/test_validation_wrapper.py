from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from effector_bincls.training.baseline import main as baseline_training_main
from effector_bincls.training.prototype_single import main as prototype_single_main
from effector_bincls.training.prototype_two_stage import (
    main as prototype_two_stage_main,
)

from .conftest import (
    copy_binary_dataset,
    create_embedding_dir,
    latest_run_dir,
    make_baseline_config,
    make_prototype_single_config,
    make_prototype_two_stage_config,
    write_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "run_validation.sh"


def run_validation_script(run_dir: Path, dataset_path: Path, *extra_args: str) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    subprocess.run(
        ["bash", str(VALIDATION_SCRIPT), str(run_dir), str(dataset_path), *extra_args],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


@pytest.mark.smoke
def test_validation_wrapper_runs_baseline_evaluation(monkeypatch, tmp_path) -> None:
    dataset_path = copy_binary_dataset(tmp_path)
    embedding_dir = create_embedding_dir(tmp_path)
    config_path = write_config(
        make_baseline_config(tmp_path, dataset_path, embedding_dir),
        tmp_path / "baseline.yml",
    )

    monkeypatch.setattr(sys, "argv", ["train-baseline", "--config", str(config_path)])
    baseline_training_main()

    run_dir = latest_run_dir(tmp_path / "baseline_results", "simple_predictor")
    run_validation_script(run_dir, dataset_path)

    assert (run_dir / "test_evaluation.yaml").exists()


@pytest.mark.smoke
def test_validation_wrapper_runs_single_stage_prototype_evaluation(
    monkeypatch,
    tmp_path,
) -> None:
    dataset_path = copy_binary_dataset(tmp_path)
    embedding_dir = create_embedding_dir(tmp_path)
    config_path = write_config(
        make_prototype_single_config(tmp_path, dataset_path, embedding_dir),
        tmp_path / "prototype_single.yml",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train-prototype-single", "--config", str(config_path)],
    )
    prototype_single_main()

    run_dir = latest_run_dir(tmp_path / "prototype_single_results", "simple")
    run_validation_script(run_dir, dataset_path, "--single-stage")

    with (run_dir / "test_evaluation.yaml").open() as handle:
        results = yaml.safe_load(handle)

    assert results["training_type"] == "single_stage"


@pytest.mark.smoke
def test_validation_wrapper_runs_two_stage_prototype_evaluation(
    monkeypatch,
    tmp_path,
) -> None:
    dataset_path = copy_binary_dataset(tmp_path)
    embedding_dir = create_embedding_dir(tmp_path)
    config_path = write_config(
        make_prototype_two_stage_config(tmp_path, dataset_path, embedding_dir),
        tmp_path / "prototype_two_stage.yml",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train-prototype-two-stage", "--config", str(config_path)],
    )
    prototype_two_stage_main()

    run_dir = latest_run_dir(tmp_path / "prototype_two_stage_results", "simple")
    run_validation_script(run_dir, dataset_path)

    with (run_dir / "test_evaluation.yaml").open() as handle:
        results = yaml.safe_load(handle)

    assert results["training_type"] == "two_stage"
