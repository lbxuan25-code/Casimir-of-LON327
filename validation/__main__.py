"""Unified command-line entry point for active validation workflows."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_COMMANDS: dict[tuple[str, str], str] = {
    ("ward", "commensurate"): "validation.commands.ward.commensurate",
    ("ward", "bond-metric-full-kernel"): (
        "validation.commands.ward.bond_metric_full_kernel"
    ),
    ("ward", "bond-metric-family"): (
        "validation.commands.ward.bond_metric_family"
    ),
    ("static", "nk-scan"): "validation.commands.static.nk_scan",
    ("static", "dwave"): "validation.commands.static.dwave_gauss_outer",
    ("static", "dwave-orbit"): "validation.commands.static.dwave_orbit_gauss",
    ("static", "projection-scan"): "validation.commands.static.projection_scan",
    ("static", "quadrature-compare"): "validation.commands.static.quadrature_compare",
    ("matsubara", "positive-point"): "validation.commands.matsubara.positive_point",
    ("matsubara", "dwave-small-xi"): "validation.commands.matsubara.dwave_small_xi",
    ("matsubara", "bond-metric-positive"): (
        "validation.commands.matsubara.bond_metric_positive"
    ),
    ("matsubara", "dwave-orbit-adaptive"): (
        "validation.commands.matsubara.dwave_orbit_adaptive"
    ),
    ("matsubara", "dwave-orbit-panel-adaptive"): (
        "validation.commands.matsubara.dwave_orbit_panel_adaptive"
    ),
    ("matsubara", "dwave-orbit-evaluator-profile"): (
        "validation.commands.matsubara.dwave_orbit_evaluator_profile"
    ),
    ("matsubara", "dwave-orbit-integrand-profile"): (
        "validation.commands.matsubara.dwave_orbit_integrand_profile"
    ),
    ("matsubara", "dwave-diagonal-width-scan"): (
        "validation.commands.matsubara.dwave_diagonal_width_scan"
    ),
    ("matsubara", "total-orbit-timing-profile"): (
        "validation.commands.matsubara.orbit_gauss_timing_profile"
    ),
    ("matsubara", "dwave-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.dwave_orbit_gauss_crosscheck"
    ),
    # Historical names remain stable; the implementation now supports exact n=0 too.
    ("matsubara", "positive-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.positive_orbit_gauss_crosscheck"
    ),
    ("matsubara", "matsubara-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.positive_orbit_gauss_crosscheck"
    ),
    # Both public staged-scan names resolve to the same total-Matsubara policy.
    ("matsubara", "positive-orbit-gauss-scan"): (
        "validation.commands.matsubara.total_orbit_gauss_scan"
    ),
    ("matsubara", "total-orbit-gauss-scan"): (
        "validation.commands.matsubara.total_orbit_gauss_scan"
    ),
    ("matsubara", "orbit-gauss-preflight"): (
        "validation.commands.matsubara.orbit_gauss_preflight"
    ),
    ("matsubara", "dwave-orbit-certification-scan"): (
        "validation.commands.matsubara.dwave_orbit_certification_scan_parallel"
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
