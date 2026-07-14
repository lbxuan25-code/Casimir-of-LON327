"""Unified command-line entry point for active validation workflows."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


# Public surface used by the pre-outer-integration main flow.  Historical positive-only
# aliases and superseded d-wave quadrature experiments are intentionally not exposed
# here; their implementation modules remain importable for tests and forensic work.
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
    ("matsubara", "total-orbit-timing-profile"): (
        "validation.commands.matsubara.orbit_gauss_timing_profile"
    ),
    ("matsubara", "matsubara-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.positive_orbit_gauss_crosscheck"
    ),
    ("matsubara", "total-orbit-gauss-scan"): (
        "validation.commands.matsubara.total_orbit_gauss_scan"
    ),
    ("matsubara", "orbit-gauss-preflight"): (
        "validation.commands.matsubara.orbit_gauss_preflight"
    ),
    # Diagnostic-only routes are deliberately separated from the outer-integration
    # intake surface.  They may localize or reproduce a blocker but never authorize
    # production input by themselves.
    ("diagnostic", "dwave-small-xi"): (
        "validation.commands.matsubara.dwave_small_xi"
    ),
    ("diagnostic", "bond-metric-positive"): (
        "validation.commands.matsubara.bond_metric_positive"
    ),
    ("diagnostic", "dwave-orbit-adaptive"): (
        "validation.commands.matsubara.dwave_orbit_adaptive"
    ),
    ("diagnostic", "dwave-orbit-panel-adaptive"): (
        "validation.commands.matsubara.dwave_orbit_panel_adaptive"
    ),
    ("diagnostic", "dwave-orbit-evaluator-profile"): (
        "validation.commands.matsubara.dwave_orbit_evaluator_profile"
    ),
    ("diagnostic", "dwave-orbit-integrand-profile"): (
        "validation.commands.matsubara.dwave_orbit_integrand_profile"
    ),
    ("diagnostic", "dwave-diagonal-width-scan"): (
        "validation.commands.matsubara.dwave_diagonal_width_scan"
    ),
    ("diagnostic", "dwave-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.dwave_orbit_gauss_crosscheck"
    ),
    ("diagnostic", "dwave-orbit-certification-scan"): (
        "validation.commands.matsubara.dwave_orbit_certification_scan_parallel"
    ),
}


def available_commands() -> tuple[tuple[str, str], ...]:
    """Return the stable public command names exposed by ``python -m validation``."""

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
    if group == "diagnostic":
        print("Diagnostic-only commands (never production authorization):")
    else:
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
