from __future__ import annotations

from validation.scripts.bdg_finite_q.finite_q_ward_scan import run_finite_q_ward_scan


def test_shifted_mesh_average_scan_option_smoke():
    report = run_finite_q_ward_scan(
        ("spm",),
        model_name="symmetry_bdg_2band",
        omega_eV=0.01,
        q_values=(0.01,),
        nk=3,
        q0_status={"spm": "diagnostic_only_not_rerun"},
        average_shifted_meshes=True,
        shift_fractions=(0.0, 0.5),
    )

    payload = report.to_dict()
    chain = payload["integrated_ward_chains"][0]
    shift_payload = chain["shifted_mesh_average"]

    assert payload["shifted_mesh_average"]["enabled"] is True
    assert payload["shifted_mesh_average"]["num_shifted_meshes"] == 4
    assert payload["shifted_mesh_average"]["direct_quadrature"] == "center"
    assert payload["rows"]
    assert payload["collective_ward_blocks"]
    assert payload["integrated_ward_chains"]
    assert shift_payload["enabled"] is True
    assert len(shift_payload["per_shift_summary"]) == 4
    assert "shift_distribution" in chain
    assert "equal_time_to_contact_difference" in chain["shift_distribution"]
    assert "component_statistics" in chain["shift_distribution"]["equal_time_to_contact_difference"]
    assert "amplitude_phase_schur_ward_distribution" in shift_payload
    assert "schur_noncommutativity" in shift_payload
    assert payload["valid_for_casimir_input"] is False


def test_endpoint_average_direct_quadrature_option_smoke():
    report = run_finite_q_ward_scan(
        ("spm",),
        model_name="symmetry_bdg_2band",
        omega_eV=0.01,
        q_values=(0.01,),
        nk=3,
        q0_status={"spm": "diagnostic_only_not_rerun"},
        average_shifted_meshes=True,
        shift_fractions=(0.0, 0.5),
        direct_quadrature="endpoint_average",
    )

    payload = report.to_dict()
    chain = payload["integrated_ward_chains"][0]
    shift_payload = chain["shifted_mesh_average"]

    assert payload["shifted_mesh_average"]["direct_quadrature"] == "endpoint_average"
    assert shift_payload["direct_quadrature"] == "endpoint_average"
    assert shift_payload["per_shift_summary"][0]["direct_quadrature"] == "endpoint_average"
    assert payload["rows"]
    assert payload["valid_for_casimir_input"] is False


def test_ward_conservative_direct_quadrature_option_smoke():
    report = run_finite_q_ward_scan(
        ("spm",),
        model_name="symmetry_bdg_2band",
        omega_eV=0.01,
        q_values=(0.01,),
        nk=3,
        q0_status={"spm": "diagnostic_only_not_rerun"},
        average_shifted_meshes=True,
        shift_fractions=(0.0, 0.5),
        direct_quadrature="ward_conservative",
    )

    payload = report.to_dict()
    chain = payload["integrated_ward_chains"][0]
    shift_payload = chain["shifted_mesh_average"]
    first_summary = shift_payload["per_shift_summary"][0]
    diagnostic = first_summary["direct_quadrature_diagnostic"]

    assert payload["shifted_mesh_average"]["direct_quadrature"] == "ward_conservative"
    assert shift_payload["direct_quadrature"] == "ward_conservative"
    assert first_summary["direct_quadrature"] == "ward_conservative"
    assert diagnostic["mode"] == "ward_conservative"
    assert "correction_norm_to_raw_direct_norm" in diagnostic
    assert payload["rows"]
    assert payload["valid_for_casimir_input"] is False
