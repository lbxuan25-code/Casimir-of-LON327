from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from lno327.conductivity import KuboConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_12_kubo_fermion_loop_sign_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_12_kubo_sign_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_positive_and_negative_bubble_helpers_are_opposite():
    module = _load_module()
    energies_minus = np.array([-0.2, 0.3])
    energies_plus = np.array([-0.1, 0.4])
    states_minus = np.eye(2, dtype=complex)
    states_plus = np.eye(2, dtype=complex)
    observable_vertices = (np.array([[1.0, 0.2], [0.2, 0.5]], dtype=complex),)
    source_vertices = (np.array([[0.3, 0.1j], [-0.1j, 0.7]], dtype=complex),)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    positive = module.finite_q_band_bubble_imag_axis_with_sign(
        energies_minus,
        states_minus,
        energies_plus,
        states_plus,
        observable_vertices,
        source_vertices,
        config,
        bubble_sign=1.0,
    )
    negative = module.finite_q_band_bubble_imag_axis_with_sign(
        energies_minus,
        states_minus,
        energies_plus,
        states_plus,
        observable_vertices,
        source_vertices,
        config,
        bubble_sign=-1.0,
    )

    np.testing.assert_allclose(positive, -negative)


def test_ward_sign_comparison_fields_are_present():
    module = _load_module()

    data = module.run_audit(q_scales=[1.0], mesh_size=8)

    assert data["stage"] == "Stage 4.12"
    assert len(data["ward_bubble_sign_audit"]["results"]) == 2
    required = {
        "R_bubble_negative",
        "R_bubble_positive",
        "C_plus",
        "K",
        "err_negative_matches_minus_C_rel",
        "err_positive_matches_plus_C_rel",
        "ward_bubble_sign_status",
    }
    for row in data["ward_bubble_sign_audit"]["results"]:
        assert required.issubset(row)


def test_positive_bubble_gives_plus_c_and_negative_gives_minus_c_on_small_mesh():
    module = _load_module()

    data = module.run_audit(q_scales=[1.0], mesh_size=8)

    for row in data["ward_bubble_sign_audit"]["results"]:
        assert row["err_positive_matches_plus_C_rel"] < 1e-8
        assert row["err_negative_matches_minus_C_rel"] < 1e-8
        assert row["ward_bubble_sign_status"] == "POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C"


def test_compressibility_sanity_check_supports_positive_sign():
    module = _load_module()

    result = module.compressibility_sanity_check()

    assert result["compressibility_status"] == "POSITIVE_BUBBLE_SIGN_MATCHES_COMPRESSIBILITY"
    assert result["analytic_compressibility"] < 0.0
    assert result["negative_bubble_static_limit"] > 0.0


def test_boundary_fields_mark_diagnostic_only_scope():
    module = _load_module()

    data = module.run_audit(q_scales=[1.0], mesh_size=8)
    boundary = data["boundary"]

    assert boundary["no_main_response_change"] is True
    assert boundary["no_direct_contact_change"] is True
    assert boundary["no_conductivity_reflection_casimir"] is True
    assert boundary["does_not_claim_ward_closure"] is True
