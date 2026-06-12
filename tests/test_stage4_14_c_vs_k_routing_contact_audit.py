from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_14_c_vs_k_routing_contact_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_14_c_vs_k_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hamiltonian_representation_check_exists_and_matches():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[8], q_scales=[1.0], temperature_sweep_K=[30.0])

    assert "hamiltonian_representation_check" in data
    assert data["diagnostic_status"]["H_representation_status"] == "H_REPRESENTATION_MATCH"
    assert (
        data["hamiltonian_representation_check"]["H_representation_status"]
        == "H_REPRESENTATION_MATCH"
    )


def test_second_order_identity_check_exists_and_matches():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[8], q_scales=[1.0], temperature_sweep_K=[30.0])

    assert "second_order_identity_check" in data
    assert data["diagnostic_status"]["second_order_identity_status"] == "SECOND_ORDER_IDENTITY_MATCH"
    assert data["second_order_identity_check"]["second_order_identity_status"] == "SECOND_ORDER_IDENTITY_MATCH"


def test_baseline_contains_c_k_and_delta_v_fields():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[8], q_scales=[1.0], temperature_sweep_K=[30.0])
    row = data["baseline_CK_results"][0]

    for key in (
        "C",
        "K_midpoint_contact",
        "K_deltaV_midpoint",
        "C_minus_K",
        "K_contact_minus_K_deltaV",
        "mesh_shift_error_rel",
    ):
        assert key in row


def test_k_contact_equals_k_delta_v_in_thermal_trace():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[8], q_scales=[1.0], temperature_sweep_K=[30.0])

    for row in data["baseline_CK_results"]:
        assert row["K_contact_minus_K_deltaV_rel"] < 1e-8


def test_boundary_fields_mark_diagnostic_only_scope():
    module = _load_module()

    data = module.run_audit(mesh_sizes=[8], q_scales=[1.0], temperature_sweep_K=[30.0])
    boundary = data["boundary"]

    assert boundary["no_main_response_change"] is True
    assert boundary["no_bubble_sign_change"] is True
    assert boundary["no_direct_contact_change"] is True
    assert boundary["no_conductivity_reflection_casimir"] is True
    assert boundary["does_not_claim_ward_closure"] is True
