from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from lno327.conductivity import KuboConfig, fermi_function, k_weights, uniform_bz_mesh
from lno327.ward_response import (
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_11_commutator_sign_quadrature_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_11_commutator_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_thermal_trace_expectation_matches_diagonal_reference():
    module = _load_module()
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )
    hamiltonian = np.diag([0.0, 1.0])
    operator = np.diag([2.0, 3.0])

    actual = module.thermal_trace_expectation(hamiltonian, operator, config)
    occupations = fermi_function(np.array([0.0, 1.0]), config.fermi_level_eV, config.temperature_eV)
    expected = np.sum(occupations * np.array([2.0, 3.0]))

    np.testing.assert_allclose(actual, expected)


def test_direct_contact_residual_equals_minus_k_on_small_mesh():
    module = _load_module()
    config = KuboConfig.from_kelvin(
        omega_eV=module.bosonic_matsubara_energy_eV(1, 30.0),
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )
    mesh = uniform_bz_mesh(4)
    weights = k_weights(mesh)
    q = np.array([0.02, 0.013])

    components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
    left_direct, _ = physical_ward_residuals(components["direct"], config.omega_eV, q)
    k_x = module.direct_contact_contraction_k(mesh, weights, config, q, "x")
    k_y = module.direct_contact_contraction_k(mesh, weights, config, q, "y")

    np.testing.assert_allclose(left_direct[1] + k_x, 0.0, atol=1e-12)
    np.testing.assert_allclose(left_direct[2] + k_y, 0.0, atol=1e-12)


def test_run_audit_smoke_and_required_fields():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[4], q_scales=[1.0])

    assert data["stage"] == "Stage 4.11"
    assert data["boundary"]["no_residual_tuning"] is True
    assert data["boundary"]["no_bubble_formula_change"] is True
    assert data["boundary"]["no_main_response_change"] is True
    assert len(data["sign_audit"]["results"]) == 2
    required = {
        "mesh_size",
        "q_scale",
        "direction_j",
        "best_bubble_sign_candidate",
        "best_bubble_sign_rel_error",
        "direct_sign_status",
        "err_Cplus_minus_K_rel",
        "err_Cminus_minus_K_rel",
        "R_total",
    }
    for row in data["sign_audit"]["results"]:
        assert required.issubset(row)
        assert row["best_bubble_sign_candidate"] in {"PLUS_C_PLUS", "MINUS_C_PLUS", "PLUS_C_MINUS", "MINUS_C_MINUS"}


def test_status_classifiers_cover_match_and_mismatch_cases():
    module = _load_module()

    assert module.classify_direct_sign(1e-12, 1.0) == "MATCH_R_DIRECT_EQUALS_MINUS_K"
    assert module.classify_direct_sign(1.0, 1e-12) == "MATCH_R_DIRECT_EQUALS_PLUS_K"
    assert module.classify_direct_sign(1e-3, 1e-2) == "MISMATCH"
    assert module.classify_convergence(1e-12, 0.0) == "NUMERICALLY_CONVERGED"
    assert module.classify_convergence(1e-4, -1.0) == "CONVERGING_WITH_MESH"
    assert module.classify_convergence(1e-4, -0.2) == "NOT_CONVERGING_OR_INCONCLUSIVE"

    consistent_rows = [
        {"mesh_size": 4, "best_bubble_sign_candidate": "PLUS_C_PLUS", "best_bubble_sign_rel_error": 1e-12},
        {"mesh_size": 8, "best_bubble_sign_candidate": "PLUS_C_PLUS", "best_bubble_sign_rel_error": 1e-12},
    ]
    assert module.classify_bubble_sign_global(consistent_rows, 8) == "CONSISTENT_MATCH_PLUS_C_PLUS"

    inconsistent_rows = [
        {"mesh_size": 4, "best_bubble_sign_candidate": "PLUS_C_PLUS", "best_bubble_sign_rel_error": 1e-12},
        {"mesh_size": 8, "best_bubble_sign_candidate": "MINUS_C_PLUS", "best_bubble_sign_rel_error": 1e-12},
    ]
    assert module.classify_bubble_sign_global(inconsistent_rows, 8) == "UNRESOLVED_OR_INCONSISTENT"


def test_stage411_output_paths_use_ward_identity_directory():
    module = _load_module()

    expected = ROOT / "validation" / "outputs" / "response" / "ward_identity"
    assert module.OUTPUT_DIR == expected
    assert module.JSON_OUTPUT.parent == expected
    assert module.MD_OUTPUT.parent == expected
