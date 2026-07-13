from __future__ import annotations

import importlib

from validation.__main__ import available_commands, resolve_command


def test_validation_cli_exposes_grouped_commands():
    expected = {
        ("ward", "commensurate"),
        ("ward", "bond-metric-full-kernel"),
        ("ward", "bond-metric-family"),
        ("static", "nk-scan"),
        ("static", "dwave"),
        ("static", "dwave-orbit"),
        ("static", "projection-scan"),
        ("static", "quadrature-compare"),
        ("matsubara", "positive-point"),
        ("matsubara", "dwave-small-xi"),
        ("matsubara", "bond-metric-positive"),
        ("matsubara", "dwave-orbit-adaptive"),
        ("matsubara", "dwave-orbit-adaptive-gk21"),
        ("matsubara", "dwave-orbit-panel-adaptive"),
        ("matsubara", "dwave-orbit-evaluator-profile"),
        ("matsubara", "dwave-orbit-gauss-crosscheck"),
        ("matsubara", "dwave-orbit-certification-scan"),
    }
    assert set(available_commands()) == expected


def test_validation_cli_command_modules_are_importable_and_callable():
    for group, command in available_commands():
        module = importlib.import_module(resolve_command(group, command))
        assert callable(module.main)
