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
    assert payload["shifted_mesh_average"]["enabled"] is True
    assert payload["shifted_mesh_average"]["num_shifted_meshes"] == 4
    assert payload["rows"]
    assert payload["collective_ward_blocks"]
    assert payload["integrated_ward_chains"]
    assert payload["integrated_ward_chains"][0]["shifted_mesh_average"]["enabled"] is True
    assert payload["valid_for_casimir_input"] is False
