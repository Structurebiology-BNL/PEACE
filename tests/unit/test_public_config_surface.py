from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "src" / "configs"
PUBLIC_DOCS = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "docs").glob("*.md")),
]

EXPECTED_PUBLIC_CONFIGS = {
    "baseline_bce.yaml",
    "contrastive_bce.yaml",
    "prototype_single_stage.yaml",
    "prototype_two_stage.yaml",
}
REMOVED_CONFIG_NAMES = {
    "baseline_simple.yaml",
    "prototype_ranking_simple.yaml",
    "prototype_ranking_two_stage_simple.yaml",
}


def test_public_config_directory_contains_only_method_named_configs() -> None:
    config_names = {path.name for path in CONFIG_DIR.glob("*.yaml")}

    assert config_names == EXPECTED_PUBLIC_CONFIGS


def test_public_docs_reference_only_existing_configs() -> None:
    docs_text = "\n".join(path.read_text() for path in PUBLIC_DOCS)

    for removed_name in REMOVED_CONFIG_NAMES:
        assert removed_name not in docs_text

    referenced_config_names = set(
        re.findall(r"src/configs/([A-Za-z0-9_.-]+\.yaml)", docs_text)
    )
    assert referenced_config_names
    assert referenced_config_names <= EXPECTED_PUBLIC_CONFIGS
    for config_name in referenced_config_names:
        assert (CONFIG_DIR / config_name).exists()


def test_public_docs_document_contrastive_bce_workflow() -> None:
    docs_text = "\n".join(path.read_text() for path in PUBLIC_DOCS)

    assert "train-contrastive-bce" in docs_text
    assert "src/configs/contrastive_bce.yaml" in docs_text
