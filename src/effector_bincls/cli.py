"""Unified package CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from effector_bincls.analysis.baseline import main as analyze_baseline_main
from effector_bincls.analysis.prototype import main as analyze_prototype_main
from effector_bincls.evaluation.baseline import main as evaluate_baseline_main
from effector_bincls.evaluation.prototype import main as evaluate_prototype_main
from effector_bincls.inference.prototype import main as infer_prototype_main
from effector_bincls.training.baseline import main as train_baseline_main
from effector_bincls.training.prototype_single import (
    main as train_prototype_single_main,
)
from effector_bincls.training.prototype_two_stage import (
    main as train_prototype_two_stage_main,
)

COMMANDS: dict[str, tuple[Callable[[], None], str]] = {
    "train-baseline": (
        train_baseline_main,
        "Train the BCE baseline workflow",
    ),
    "train-prototype-single": (
        train_prototype_single_main,
        "Train the single-stage prototype contrastive workflow",
    ),
    "train-prototype-two-stage": (
        train_prototype_two_stage_main,
        "Train the two-stage prototype contrastive workflow",
    ),
    "evaluate-baseline": (
        evaluate_baseline_main,
        "Evaluate a baseline run directory on a test CSV",
    ),
    "evaluate-prototype": (
        evaluate_prototype_main,
        "Evaluate a prototype run directory on a test CSV",
    ),
    "infer-prototype": (
        infer_prototype_main,
        "Run prototype inference on a saved run directory",
    ),
    "analyze-baseline": (
        analyze_baseline_main,
        "Generate baseline analysis artifacts for a saved run",
    ),
    "analyze-prototype": (
        analyze_prototype_main,
        "Generate prototype analysis artifacts for a saved run",
    ),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the requested package subcommand."""
    parser = argparse.ArgumentParser(prog="effector-bincls")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(command, help=help_text, add_help=False)

    args, remaining = parser.parse_known_args(argv)
    handler, _ = COMMANDS[args.command]
    sys.argv = [args.command, *remaining]
    handler()


if __name__ == "__main__":
    main()
