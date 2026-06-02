"""Shared pytest classification for the research test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

BENCHMARK_KEYWORDS = (
    "benchmark",
    "convergence",
    "distance_scan",
    "high_nk",
    "sensitive_sampling",
    "adaptive_integration",
    "n0_torque",
)

DIAGNOSTIC_KEYWORDS = (
    "finite_q",
    "static_policy",
    "superconducting_response",
    "paramagnetic_kernel_diagnosis",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        filename = Path(str(item.fspath)).name
        stem = filename.removeprefix("test_").removesuffix(".py")

        if any(keyword in stem for keyword in BENCHMARK_KEYWORDS):
            item.add_marker(pytest.mark.benchmark)
        if any(keyword in stem for keyword in DIAGNOSTIC_KEYWORDS):
            item.add_marker(pytest.mark.diagnostic)
        if "finite_q" in stem or any(keyword in stem for keyword in BENCHMARK_KEYWORDS):
            item.add_marker(pytest.mark.regression)
        if not any(mark.name in {"benchmark", "diagnostic", "regression"} for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
