from __future__ import annotations

import importlib

from validation.__main__ import available_commands, resolve_command


def test_validation_cli_exposes_tight_main_and_diagnostic_surfaces():
    expected = {
        ("ward", "commensurate"),
        ("ward", "bond-metric-full-kernel"),
        ("ward", "bond-metric-family"),
        ("matsubara", "total-orbit-timing-profile"),
        ("matsubara", "matsubara-orbit-gauss-crosscheck"),
        ("matsubara", "total-orbit-gauss-scan"),
        ("matsubara", "orbit-gauss-preflight"),
        ("matsubara", "arbitrary-q-performance-preflight"),
        ("matsubara", "arbitrary-q-periodic-bz-qualification"),
        ("casimir", "outer-q-quadrature-preflight"),
        ("casimir", "microscopic-outer-q-preflight"),
        ("diagnostic", "arbitrary-q-performance-smoke"),
        ("diagnostic", "arbitrary-q-physics-smoke"),
        ("diagnostic", "transverse-point-sweet-spot"),
    }
    assert set(available_commands()) == expected


def test_unified_sweet_spot_is_only_public_fixed_point_transverse_route():
    commands = set(available_commands())
    assert ("diagnostic", "transverse-point-sweet-spot") in commands
    for retired in (
        ("matsubara", "positive-point"),
        ("static", "nk-scan"),
        ("static", "dwave"),
        ("static", "dwave-orbit"),
        ("static", "projection-scan"),
        ("static", "quadrature-compare"),
        ("diagnostic", "arbitrary-q-uniform-refinement"),
        ("diagnostic", "dwave-small-xi"),
        ("diagnostic", "bond-metric-positive"),
        ("diagnostic", "dwave-orbit-adaptive"),
        ("diagnostic", "dwave-orbit-panel-adaptive"),
        ("diagnostic", "dwave-orbit-evaluator-profile"),
        ("diagnostic", "dwave-orbit-integrand-profile"),
        ("diagnostic", "dwave-diagonal-width-scan"),
        ("diagnostic", "dwave-orbit-gauss-crosscheck"),
        ("diagnostic", "dwave-orbit-certification-scan"),
    ):
        assert retired not in commands


def test_historical_positive_only_aliases_are_not_public():
    commands = set(available_commands())
    assert ("matsubara", "positive-orbit-gauss-crosscheck") not in commands
    assert ("matsubara", "positive-orbit-gauss-scan") not in commands
    assert ("matsubara", "dwave-orbit-gauss-crosscheck") not in commands


def test_validation_cli_command_modules_are_importable_and_callable():
    for group, command in available_commands():
        module = importlib.import_module(resolve_command(group, command))
        assert callable(module.main)
