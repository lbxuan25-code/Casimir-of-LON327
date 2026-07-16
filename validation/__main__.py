"""Unified command-line entry point for active validation workflows."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


# Public validation surface for response, point integration, and staged outer
# Casimir integration qualification.
_COMMANDS: dict[tuple[str, str], str] = {
    ("ward", "commensurate"): "validation.commands.ward.commensurate",
    ("ward", "bond-metric-full-kernel"): (
        "validation.commands.ward.bond_metric_full_kernel"
    ),
    ("ward", "bond-metric-family"): (
        "validation.commands.ward.bond_metric_family"
    ),
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
    ("matsubara", "arbitrary-q-performance-preflight"): (
        "validation.commands.matsubara.arbitrary_q_performance_preflight"
    ),
    ("matsubara", "arbitrary-q-periodic-bz-qualification"): (
        "validation.commands.matsubara.arbitrary_q_periodic_bz_qualification_gate"
    ),
    ("casimir", "outer-q-quadrature-preflight"): (
        "validation.commands.casimir.outer_q_quadrature_preflight"
    ),
    ("casimir", "microscopic-outer-q-preflight"): (
        "validation.commands.casimir.microscopic_outer_q_preflight"
    ),
    # The only public fixed-point transverse-integration command. It handles both
    # pairings, zero/positive Matsubara indices, arbitrary q directions and
    # point-specific N/shift sweet-spot selection.
    ("diagnostic", "transverse-point-sweet-spot"): (
        "validation.commands.matsubara.transverse_point_sweet_spot"
    ),
    ("diagnostic", "arbitrary-q-performance-smoke"): (
        "validation.commands.matsubara.arbitrary_q_performance_smoke"
    ),
    ("diagnostic", "arbitrary-q-physics-smoke"): (
        "validation.commands.matsubara.arbitrary_q_physics_smoke"
    ),
}

# Hidden compatibility routes retain only names needed by aggregate orchestration.
# Removed single-point commands have no public or hidden runnable aliases.
_INTERNAL_ALIASES: dict[tuple[str, str], str] = {
    ("matsubara", "positive-orbit-gauss-crosscheck"): (
        "validation.commands.matsubara.positive_orbit_gauss_crosscheck"
    ),
    ("matsubara", "positive-orbit-gauss-scan"): (
        "validation.commands.matsubara.total_orbit_gauss_scan"
    ),
}


def available_commands() -> tuple[tuple[str, str], ...]:
    """Return the stable public command names exposed by ``python -m validation``."""
    return tuple(sorted(_COMMANDS))


def resolve_command(group: str, command: str) -> str:
    """Resolve a public command or a narrowly retained internal alias."""
    key = (str(group), str(command))
    if key in _COMMANDS:
        return _COMMANDS[key]
    if key in _INTERNAL_ALIASES:
        return _INTERNAL_ALIASES[key]
    raise ValueError(f"unknown validation command: {group} {command}")


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
