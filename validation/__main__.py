"""Unified command-line entry point for active validation workflows."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_COMMANDS: dict[tuple[str, str], str] = {
    ("ward", "contract-audit"): "validation.commands.ward.contract_audit",
    ("ward", "commensurate"): "validation.commands.ward.commensurate",
    ("ward", "phase-column"): "validation.commands.ward.phase_column",
    ("ward", "phase-hessian"): "validation.commands.ward.phase_hessian",
    ("ward", "phase-hessian-family"): "validation.commands.ward.phase_hessian_family",
    ("ward", "average-subgrids"): "validation.commands.ward.average_subgrids",
    ("ward", "bond-metric-full-kernel"): (
        "validation.commands.ward.bond_metric_full_kernel_fast"
    ),
    ("ward", "bond-metric-family"): (
        "validation.commands.ward.bond_metric_family_fast"
    ),
    ("static", "nk-scan"): "validation.commands.static.nk_scan",
    ("static", "dwave"): "validation.commands.static.dwave_gauss_outer",
    ("static", "projection-scan"): "validation.commands.static.projection_scan",
    ("static", "quadrature-compare"): "validation.commands.static.quadrature_compare",
    ("matsubara", "positive-point"): "validation.commands.matsubara.positive_point",
    ("matsubara", "dwave-small-xi"): "validation.commands.matsubara.dwave_small_xi",
    ("matsubara", "bond-metric-positive"): (
        "validation.commands.matsubara.bond_metric_positive"
    ),
}


def available_commands() -> tuple[tuple[str, str], ...]:
    """Return the stable command names exposed by ``python -m validation``."""

    return tuple(sorted(_COMMANDS))


def resolve_command(group: str, command: str) -> str:
    """Resolve one public command pair to its implementation module."""

    try:
        return _COMMANDS[(str(group), str(command))]
    except KeyError as exc:
        raise ValueError(f"unknown validation command: {group} {command}") from exc


def _print_help(group: str | None = None) -> None:
    print("usage: python -m validation <group> <command> [options]")
    print("")
    print("Active validation commands:")
    for current_group, command in available_commands():
        if group is None or current_group == group:
            print(f"  {current_group} {command}")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return
    if len(args) == 1 or args[1] in {"-h", "--help"}:
        _print_help(args[0])
        return

    group, command = args[0], args[1]
    try:
        module_name = resolve_command(group, command)
    except ValueError as exc:
        _print_help(group)
        raise SystemExit(str(exc)) from exc

    module = importlib.import_module(module_name)
    command_main = getattr(module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"validation command module has no callable main(): {module_name}")

    original_argv = sys.argv
    try:
        sys.argv = [f"python -m validation {group} {command}", *args[2:]]
        command_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
