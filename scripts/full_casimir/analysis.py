"""Post-processing command handlers dispatched only by ``scripts.full_casimir``."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSTPROCESS_ROOT,
    DEFAULT_SCAN_STEP_DEG,
)
from .plotting import plot_results
from .postprocess import postprocess_torque


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir",
        description="Post-process an existing formal Casimir campaign.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    torque = commands.add_parser("torque")
    plot = commands.add_parser("plot")
    for command in (torque, plot):
        command.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument(
            "--postprocess-root",
            type=Path,
            default=DEFAULT_POSTPROCESS_ROOT,
        )
        command.add_argument("--profile", default=None)
        command.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)
    return parser


def _run_torque(args: argparse.Namespace) -> int:
    energy_csv, torque_csv, metadata, complete = postprocess_torque(
        run_root=args.run_root,
        output_root=args.postprocess_root,
        profile=args.profile,
        step_deg=args.angle_step,
    )
    for path in (energy_csv, torque_csv, metadata):
        print(f"written: {path}")
    return 0 if complete else 2


def _run_plot(args: argparse.Namespace) -> int:
    outputs = plot_results(
        output_root=args.postprocess_root,
        profile=args.profile,
    )
    for path in outputs:
        print(f"written: {path}")
    return 0 if outputs else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "torque":
        return _run_torque(args)
    return _run_plot(args)


__all__ = ["main"]
