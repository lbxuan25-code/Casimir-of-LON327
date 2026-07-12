from __future__ import annotations

import importlib

from validation.__main__ import available_commands, resolve_command


def test_validation_cli_exposes_grouped_commands():
    expected = {
        ("ward", "contract-audit"),
        ("ward", "commensurate"),
        ("ward", "phase-column"),
        ("ward", "phase-hessian"),
        ("ward", "phase-hessian-family"),
        ("ward", "average-subgrids"),
        ("ward", "bond-metric-full-kernel"),
        ("static", "nk-scan"),
        ("static", "projection-scan"),
        ("static", "quadrature-compare"),
        ("static", "dwave-gauss-outer"),
        ("static", "dwave-iterated-adaptive"),
        ("static", "dwave-shift-batch"),
        ("static", "dwave-shift-budget"),
        ("static", "dwave-shift-reference"),
        ("matsubara", "positive-point"),
        ("matsubara", "dwave-small-xi"),
    }
    assert set(available_commands()) == expected


def test_validation_cli_command_modules_are_importable_and_callable():
    for group, command in available_commands():
        module = importlib.import_module(resolve_command(group, command))
        assert callable(module.main)
