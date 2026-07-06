from __future__ import annotations

import pytest

from validation.lib.finite_q_schur_ward_localization import run_bdg_schur_ward_algebra_localization


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
def test_schur_ward_analytic_identity_payload(pairing_name):
    payload = run_bdg_schur_ward_algebra_localization(pairing_name=pairing_name)
    identity = payload["analytic_identity"]

    assert payload["valid_for_casimir_input"] is False
    assert identity["valid_for_casimir_input"] is False
    assert identity["eta2_relation"] == "eta2 = delta0 * theta"
    assert identity["source_convention"] == "endpoint_average_delta_minus_plus_delta_plus_over_2delta0"
    assert identity["aa_identity"] == "full_hessian"
    assert identity["aa_kernel_definition"] == "K_AA_full = K_AA_bubble + K_AA_direct"
    assert identity["direct_role"] == "included_in_full_hessian_not_subtracted_from_formal_identity"
    assert identity["classification"] in {
        "aa_small_mixed_small",
        "aa_small_mixed_large",
        "aa_large_mixed_small",
        "aa_large_mixed_large",
    }
    assert set(identity) >= {
        "analytic_R_left",
        "analytic_R_right",
        "full_hessian_left_aa_norm",
        "full_hessian_right_aa_norm",
        "homogeneous_left_aa_norm",
        "homogeneous_right_aa_norm",
        "left_aeta_norm",
        "right_etaa_norm",
        "max_full_hessian_norm",
        "classification",
        "valid_for_casimir_input",
    }


@pytest.mark.parametrize("forbidden", [
    "contact_aware_left_aa_norm",
    "contact_aware_right_aa_norm",
    "max_contact_aware_norm",
    "max_bubble_only_diagnostic_norm",
])
def test_schur_ward_analytic_identity_does_not_export_contact_aware_fields(forbidden):
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    identity = payload["analytic_identity"]

    assert forbidden not in identity


def test_schur_ward_analytic_generators_are_fixed_by_delta0():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave", delta0_eV=0.1)
    identity = payload["analytic_identity"]

    left = identity["analytic_R_left"]
    right = identity["analytic_R_right"]

    assert left[0]["abs"] == pytest.approx(0.0)
    assert left[1]["real"] == pytest.approx(0.0)
    assert left[1]["imag"] == pytest.approx(0.2)
    assert right[0]["abs"] == pytest.approx(0.0)
    assert right[1]["real"] == pytest.approx(0.0)
    assert right[1]["imag"] == pytest.approx(-0.2)


def test_schur_ward_analytic_identity_uses_full_hessian_aa_norm():
    payload = run_bdg_schur_ward_algebra_localization(pairing_name="dwave")
    identity = payload["analytic_identity"]

    assert identity["max_full_hessian_norm"] == pytest.approx(
        max(
            identity["full_hessian_left_aa_norm"],
            identity["full_hessian_right_aa_norm"],
            identity["left_aeta_norm"],
            identity["right_etaa_norm"],
        )
    )
    assert identity["homogeneous_left_aa_norm"] == pytest.approx(identity["full_hessian_left_aa_norm"])
    assert identity["homogeneous_right_aa_norm"] == pytest.approx(identity["full_hessian_right_aa_norm"])
