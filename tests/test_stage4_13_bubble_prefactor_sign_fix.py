from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh
from lno327.ward_response import (
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_13_bubble_sign_fix_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_13_bubble_sign_fix", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rel_error(lhs: complex, rhs: complex) -> float:
    return float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))


def test_main_bubble_prefactor_fix_gives_plus_c_and_preserves_direct_contact():
    module = _load_module()
    mesh = uniform_bz_mesh(8)
    weights = k_weights(mesh)
    temperature_K = 30.0
    omega_eV = bosonic_matsubara_energy_eV(1, temperature_K)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=1e-10,
        output_si=False,
    )
    q = np.array([0.02, 0.013], dtype=float)

    components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
    left_bubble, _ = physical_ward_residuals(components["bubble"], omega_eV, q)
    left_direct, _ = physical_ward_residuals(components["direct"], omega_eV, q)
    left_total, _ = physical_ward_residuals(components["total"], omega_eV, q)

    for j_index, direction_j in enumerate(("x", "y")):
        c_j = module.commutator_candidate_c_plus_q(mesh, weights, config, q, direction_j)
        k_j = module.direct_contact_contraction_k(mesh, weights, config, q, direction_j)
        assert _rel_error(left_bubble[1 + j_index], c_j) < 1e-8
        assert _rel_error(left_direct[1 + j_index], -k_j) < 1e-10
        assert _rel_error(left_total[1 + j_index], c_j - k_j) < 1e-8


def test_stage413_regression_reports_expected_boundaries_and_statuses():
    module = _load_module()

    data = module.run_regression(mesh_size=8)

    assert data["stage"] == "Stage 4.13"
    assert data["diagnostic_status"]["main_bubble_sign_status"] == "MAIN_BUBBLE_MATCHES_PLUS_C"
    assert data["diagnostic_status"]["direct_contact_status"] == "DIRECT_STILL_MATCHES_MINUS_K"
    assert data["diagnostic_status"]["total_bookkeeping_status"] == "TOTAL_MATCHES_C_MINUS_K"
    boundary = data["boundary"]
    assert boundary["bubble_prefactor_changed"] is True
    assert boundary["direct_contact_unchanged"] is True
    assert boundary["source_observable_split_unchanged"] is True
    assert boundary["no_conductivity_reflection_casimir"] is True
