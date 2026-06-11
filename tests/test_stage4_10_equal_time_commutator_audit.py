from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh
from lno327.ward_response import (
    normal_physical_density_current_response_components_imag_axis,
    normal_physical_density_current_response_imag_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_49 = ROOT / "validation" / "scripts" / "response" / "stage4_9_physical_ward_residual_regression.py"
SCRIPT_410 = ROOT / "validation" / "scripts" / "response" / "stage4_10_equal_time_commutator_audit.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_projection_helper_uses_linear_spatial_projection():
    module = _load(SCRIPT_410, "stage4_10")
    q = np.array([1.0, 0.0])
    spatial = np.array([2.0 + 1.0j, 3.0 - 1.0j])

    longitudinal, transverse = module.project_spatial_components(q, spatial)

    assert longitudinal == 2.0 + 1.0j
    assert transverse == 3.0 - 1.0j


def test_second_order_peierls_identity_helper_single_point():
    module = _load(SCRIPT_410, "stage4_10")
    q = np.array([0.02, 0.013])

    sample = module.second_order_identity_sample(0.17, -0.23, q, "x")

    assert sample["abs_error"] < 1e-12
    assert sample["rel_error"] < 1e-10


def test_response_components_sum_to_total_and_main_response():
    mesh = uniform_bz_mesh(4)
    weights = k_weights(mesh)
    q = np.array([0.01, 0.007])
    config = KuboConfig.from_kelvin(
        omega_eV=bosonic_matsubara_energy_eV(1, 30.0),
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
    total = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)

    np.testing.assert_allclose(components["bubble"] + components["direct"], components["total"])
    np.testing.assert_allclose(components["total"], total)


def test_status_classifiers():
    module = _load(SCRIPT_410, "stage4_10")

    assert module.classify_second_order_identity(1e-12) == "MATCH"
    assert module.classify_second_order_identity(1e-8) == "MISMATCH"
    assert module.classify_direct_completion(1e-12, 0.0) == "DIRECT_TERM_NUMERICALLY_CLOSES_WARD"
    assert module.classify_direct_completion(1e-5, 1.0) == "DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL"
    assert module.classify_direct_completion(1e-5, 2.0) == "DIRECT_TERM_LEAVES_ORDER_Q2_OR_BETTER_RESIDUAL"
    assert module.classify_direct_completion(1e-5, 0.2) == "DIRECT_TERM_LEAVES_NONSCALING_RESIDUAL"


def test_run_audit_smoke_with_test_mesh():
    module = _load(SCRIPT_410, "stage4_10")

    data = module.run_audit(mesh_size=12)

    assert data["stage"] == "Stage 4.10"
    assert "second_order_peierls_identity" in data
    assert "residual_decomposition" in data
    assert "diagnostic_status" in data


def test_stage49_and_stage410_output_paths_use_ward_identity_directory():
    stage49 = _load(SCRIPT_49, "stage4_9")
    stage410 = _load(SCRIPT_410, "stage4_10")

    expected = ROOT / "validation" / "outputs" / "response" / "ward_identity"
    assert stage49.OUTPUT_DIR == expected
    assert stage410.OUTPUT_DIR == expected
    assert stage49.JSON_OUTPUT.parent == expected
    assert stage410.JSON_OUTPUT.parent == expected
