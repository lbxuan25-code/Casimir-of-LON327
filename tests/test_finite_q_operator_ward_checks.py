from __future__ import annotations

import numpy as np

from lno327 import uniform_bz_mesh
from validation.lib.finite_q_operator_ward_checks import evaluate_bdg_operator_ward_checks
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def test_operator_ward_check_payload_reports_bdg_matrix_identities():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params(delta0_eV=0.1)
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    payload = evaluate_bdg_operator_ward_checks(
        pairing_name="dwave",
        q_model=np.asarray([0.02, 0.0]),
        delta0_eV=0.1,
        spec=model.spec,
        ansatz=ansatz,
        amp=amp,
        k_points=uniform_bz_mesh(3),
    )

    assert payload["identity_version"] == "finite_q_bdg_operator_ward_v1"
    assert payload["diagnostic_role"] == "operator_identity_diagnostic_not_a_new_ward_criterion"
    assert payload["pairing_name"] == "dwave"
    assert payload["q_model"] == [0.02, 0.0]
    assert payload["first_order_bdg_identity"]["max_error_norm"] >= 0.0
    assert payload["bdg_contact_identity"]["max_error_norm"] >= 0.0
    assert payload["first_order_bdg_identity"]["ranked_errors"]
    assert payload["bdg_contact_identity"]["ranked_errors"]
    assert payload["valid_for_casimir_input"] is False


def test_operator_ward_check_rejects_bad_q_shape():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params(delta0_eV=0.1)
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")

    try:
        evaluate_bdg_operator_ward_checks(
            pairing_name="spm",
            q_model=np.asarray([0.01, 0.0, 0.0]),
            delta0_eV=0.1,
            spec=model.spec,
            ansatz=ansatz,
            amp=amp,
            k_points=uniform_bz_mesh(3),
        )
    except ValueError as exc:
        assert "q_model" in str(exc)
    else:
        raise AssertionError("bad q_model shape should raise ValueError")
