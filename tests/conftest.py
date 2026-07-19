"""Shared pytest classification and deterministic numerical environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The production arbitrary-q process pool deliberately fails closed unless every
# BLAS/OpenMP backend is restricted to one thread.  GitHub Actions exports this
# contract before Python starts; local pytest establishes the same contract here,
# before test modules import NumPy, so results do not depend on the caller's shell.
_SINGLE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _name in _SINGLE_THREAD_ENV:
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"


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
        if any(keyword in stem for keyword in BENCHMARK_KEYWORDS):
            item.add_marker(pytest.mark.regression)
        if not any(
            mark.name in {"benchmark", "diagnostic", "regression"}
            for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
