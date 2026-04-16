from __future__ import annotations

import sys

import pytest

from effector_bincls.analysis.baseline import main as baseline_analysis_main
from effector_bincls.analysis.prototype import main as prototype_analysis_main
from effector_bincls.evaluation.baseline import main as baseline_evaluation_main
from effector_bincls.evaluation.prototype import main as prototype_evaluation_main
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


@pytest.mark.smoke
def test_baseline_package_workflow(monkeypatch, tmp_path) -> None:
    dataset_path = copy_binary_dataset(tmp_path)
    embedding_dir = create_embedding_dir(tmp_path)
    config_path = write_config(
        make_baseline_config(tmp_path, dataset_path, embedding_dir),
        tmp_path / "baseline.yml",
    )

    monkeypatch.setattr(sys, "argv", ["train-baseline", "--config", str(config_path)])
    baseline_training_main()

    run_dir = latest_run_dir(tmp_path / "baseline_results", "simple_predictor")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-baseline",
            "--run_dir",
            str(run_dir),
            "--test_csv",
            str(dataset_path),
        ],
    )
    baseline_evaluation_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze-baseline",
            "--run_dir",
            str(run_dir),
            "--sample_size",
            "2",
        ],
    )
    baseline_analysis_main()

    assert (run_dir / "results.yaml").exists()
    assert (run_dir / "test_evaluation.yaml").exists()
    assert (run_dir / "baseline_analysis" / "baseline_analysis_summary.json").exists()


@pytest.mark.smoke
def test_single_stage_prototype_package_workflow(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-prototype",
            "--run_dir",
            str(run_dir),
            "--test_csv",
            str(dataset_path),
            "--single-stage",
        ],
    )
    prototype_evaluation_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze-prototype",
            "--run_dir",
            str(run_dir),
            "--sample_size",
            "2",
            "--single-stage",
        ],
    )
    prototype_analysis_main()

    assert (run_dir / "results.yaml").exists()
    assert (run_dir / "test_evaluation.yaml").exists()
    assert (
        run_dir / "prototype_analysis" / "analysis_summary_with_prototypes.json"
    ).exists()


@pytest.mark.smoke
def test_two_stage_prototype_package_workflow(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-prototype",
            "--run_dir",
            str(run_dir),
            "--test_csv",
            str(dataset_path),
        ],
    )
    prototype_evaluation_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze-prototype",
            "--run_dir",
            str(run_dir),
            "--sample_size",
            "2",
        ],
    )
    prototype_analysis_main()

    assert (run_dir / "results.yaml").exists()
    assert (run_dir / "test_evaluation.yaml").exists()
    assert (
        run_dir / "prototype_analysis" / "analysis_summary_with_prototypes.json"
    ).exists()
