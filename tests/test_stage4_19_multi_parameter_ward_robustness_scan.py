from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lno327.ward_response import physical_ward_residuals

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_19_multi_parameter_ward_robustness_scan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_19_ward_robustness", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lightweight_scan():
    module = _load_module()
    return module.run_scan(
        temperatures_K=[30.0],
        matsubara_indices=[1],
        q_cases={"q_diag_pos": np.array([0.02, 0.013], dtype=float)},
        q_scales=[1.0, 0.5],
        adaptive_levels=[1],
        gauss_orders=[2],
        fermi_windows_eV=[0.05],
        coarse_grid=8,
    )


def test_lightweight_run_outputs_required_top_level_fields(lightweight_scan):
    assert lightweight_scan["stage"] == "Stage 4.19"
    for key in (
        "scan_results",
        "summary_statistics",
        "worst_cases",
        "diagnostic_status",
        "boundary",
    ):
        assert key in lightweight_scan


def test_each_scan_result_contains_required_fields(lightweight_scan):
    assert lightweight_scan["scan_results"]
    required = (
        "temperature_K",
        "matsubara_index",
        "q_case",
        "q_scale",
        "adaptive_level",
        "gauss_order",
        "fermi_window_eV",
        "left_norm",
        "right_norm",
        "max_corrected_norm",
        "status",
    )
    for row in lightweight_scan["scan_results"]:
        for key in required:
            assert key in row


def test_corrected_right_residual_uses_minus_q_contraction():
    response = np.array(
        [
            [1.0 + 0.1j, 0.2 - 0.3j, -0.4 + 0.2j],
            [0.5 - 0.7j, 1.2 + 0.4j, 0.1 + 0.8j],
            [-0.2 + 0.6j, 0.3 - 0.5j, 0.9 + 0.2j],
        ],
        dtype=complex,
    )
    omega_eV = 0.02
    q = np.array([0.02, -0.013], dtype=float)
    _left, right = physical_ward_residuals(response, omega_eV, q)
    expected_right = 1j * omega_eV * response[:, 0] - q[0] * response[:, 1] - q[1] * response[:, 2]
    np.testing.assert_allclose(right, expected_right)


def test_boundary_fields_are_all_true(lightweight_scan):
    for value in lightweight_scan["boundary"].values():
        assert value is True


def test_no_downstream_reflection_or_casimir_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("reflection" in module for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)


def test_worst_cases_are_sorted_descending(lightweight_scan):
    for key in (
        "top_10_largest_max_corrected_norm",
        "top_10_largest_left_norm",
        "top_10_largest_right_norm",
    ):
        rows = lightweight_scan["worst_cases"][key]
        if "max_corrected" in key:
            values = [float(row["max_corrected_norm"]) for row in rows]
        elif "left" in key:
            values = [float(row["left_norm"]) for row in rows]
        else:
            values = [float(row["right_norm"]) for row in rows]
        assert values == sorted(values, reverse=True)


def test_summary_statistics_are_consistent(lightweight_scan):
    stats = lightweight_scan["summary_statistics"]
    assert stats["num_total_cases"] == len(lightweight_scan["scan_results"])
    assert (
        stats["num_closed"]
        + stats["num_acceptable_but_monitor"]
        + stats["num_not_closed"]
        == stats["num_total_cases"]
    )
