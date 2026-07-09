from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.contact_formula_audit import (
    SCHEMA_VERSION,
    analyze_contact_formula,
    componentwise_ratio,
    parallelism,
    scalar_projection,
    side_contact_audit,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_contact_formula_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_contact_formula_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_contact_formula_audit_v1"


def test_scalar_projection_recovers_required_over_current_factor():
    current = np.asarray([1.0 + 1.0j, 2.0, -0.5j], dtype=complex)
    alpha = 0.8268 - 0.1j
    required = alpha * current
    report = scalar_projection(required, current)
    np.testing.assert_allclose(report["alpha_required_over_current"], alpha, atol=1e-12)
    assert report["residual_norm"] < 1e-12


def test_componentwise_ratio_handles_zero_current_component():
    required = np.asarray([2.0, 3.0], dtype=complex)
    current = np.asarray([1.0, 0.0], dtype=complex)
    rows = componentwise_ratio(required, current, ("A", "B"))
    assert rows[0]["ratio_defined"] is True
    np.testing.assert_allclose(rows[0]["required_over_current"], 2.0 + 0.0j)
    assert rows[1]["ratio_defined"] is False


def test_parallelism_reports_unit_overlap_for_parallel_vectors():
    current = np.asarray([1.0, 2.0j], dtype=complex)
    required = (0.5 + 0.25j) * current
    report = parallelism(required, current)
    assert report["abs_overlap"] == pytest.approx(1.0)


def test_side_contact_audit_uses_required_equals_minus_bubble_minus_mixed():
    bubble = np.asarray([1.0, 2.0, 0.0], dtype=complex)
    mixed = np.asarray([-0.25, 0.5, 0.0], dtype=complex)
    required = -bubble - mixed
    current = 2.0 * required
    report = side_contact_audit(side="left", current=current, required=required, bubble=bubble, mixed=mixed)
    np.testing.assert_allclose(
        [item["value"] for item in report["contact_required"]["values"]],
        required,
    )
    np.testing.assert_allclose(report["required_over_current_scalar_projection"]["alpha_required_over_current"], 0.5)
    np.testing.assert_allclose(
        [item["value"] for item in report["ward_residual_with_current_contact"]["values"]],
        current - required,
    )


def test_analyze_contact_formula_constructed_left_right_consistency():
    k_ss_bubble = np.diag([1.0, 2.0, 3.0]).astype(complex)
    k_ss_contact = np.diag([10.0, 20.0, 30.0]).astype(complex)
    k_seta = np.asarray([[0.5], [0.25], [0.0]], dtype=complex)
    k_etas = np.asarray([[0.5, 0.25, 0.0]], dtype=complex)
    primitive = {
        "k_ss_bubble": k_ss_bubble,
        "k_ss_contact": k_ss_contact,
        "k_seta": k_seta,
        "k_etas": k_etas,
    }
    candidate = {
        "candidate": "constructed",
        "description": "constructed",
        "left_u": np.asarray([1.0, 0.0, 0.0], dtype=complex),
        "right_u": np.asarray([1.0, 0.0, 0.0], dtype=complex),
        "left_w": np.asarray([1.0], dtype=complex),
        "right_w": np.asarray([1.0], dtype=complex),
    }
    report = analyze_contact_formula(primitive=primitive, candidate=candidate, collective_order=("phase_eta2",))
    left_alpha = report["left"]["required_over_current_scalar_projection"]["alpha_required_over_current"]
    right_alpha = report["right"]["required_over_current_scalar_projection"]["alpha_required_over_current"]
    np.testing.assert_allclose(left_alpha, -0.15 + 0.0j)
    np.testing.assert_allclose(right_alpha, -0.15 + 0.0j)
    assert report["left_right_scalar_consistency"]["abs_difference"] < 1e-14
    assert report["accepted_convention"] is False


def test_contact_formula_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--q",
                "0.02",
                "--nk",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_contact_formula_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "-1",
                "--q",
                "0.02",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
