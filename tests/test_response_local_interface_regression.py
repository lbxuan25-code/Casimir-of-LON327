import numpy as np
import pytest

from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.numerics.grids import uniform_bz_mesh
from lno327.numerics.weights import k_weights
from lno327.response.local_interface import (
    LocalSheetResponse,
    compare_local_responses_imag_axis,
    conductivity_tensor_from_matrix,
    local_response_imag_axis as new_local_response,
    matrix_symmetry_diagnostics,
    validate_local_response_symmetry,
)
from lno327.response_interface import local_response_imag_axis as old_local_response


def _inputs():
    points = uniform_bz_mesh(2)
    return points, k_weights(points), PairingAmplitudes(delta0_eV=0.04)


def _assert_common_response_fields_match(new, old):
    np.testing.assert_allclose(new.matrix, old.matrix, rtol=1e-12, atol=1e-12)
    assert new.kind == old.kind
    assert new.omega_eV == old.omega_eV
    assert new.unit_label == old.unit_label
    assert new.valid_for_casimir_input is False
    assert old.valid_for_casimir_input is False
    assert new.normalization_status == old.normalization_status
    assert new.static_policy == old.static_policy
    assert new.momentum_status == old.momentum_status


@pytest.mark.parametrize("kind", ("normal", "spm", "dwave"))
def test_new_local_response_matches_old_reference_for_positive_omega(kind):
    points, weights, amp = _inputs()
    kwargs = {
        "kind": kind,
        "omega_eV": 0.03,
        "k_points": points,
        "temperature_K": 20.0,
        "eta_eV": 1e-4,
        "pairing_params": amp,
        "k_weights": weights,
    }

    new = new_local_response(**kwargs)
    old = old_local_response(**kwargs)

    _assert_common_response_fields_match(new, old)
    assert new.source in {
        "kubo_conductivity_imag_axis_from_model",
        "bdg_local_superconducting_response_imag_axis",
    }


def test_new_local_response_matches_old_reference_for_normal_zero_omega():
    points, weights, amp = _inputs()
    new = new_local_response("normal", 0.0, points, 20.0, pairing_params=amp, k_weights=weights)
    old = old_local_response("normal", 0.0, points, 20.0, pairing_params=amp, k_weights=weights)

    _assert_common_response_fields_match(new, old)
    assert new.source == "kubo_conductivity_imag_axis_from_model"
    assert new.static_policy == "n0_unresolved"


def test_new_local_response_rejects_bdg_zero_omega_like_old_reference():
    points, _, amp = _inputs()
    for local_response in (new_local_response, old_local_response):
        with pytest.raises(ValueError, match="n=0 is unresolved"):
            local_response("spm", 0.0, points, 20.0, pairing_params=amp)


def test_local_interface_helpers_preserve_public_behavior():
    matrix = np.array([[2.0 + 0.0j, 0.0], [0.0, 2.0 + 0.0j]], dtype=complex)
    tensor = conductivity_tensor_from_matrix(matrix)
    assert tensor.matrix().shape == (2, 2)
    np.testing.assert_allclose(tensor.matrix(), matrix)

    response = LocalSheetResponse(
        kind="normal",
        omega_eV=0.1,
        matrix=matrix,
        unit_label="test",
        source="test",
        valid_for_casimir_input=False,
        notes=(),
    )
    assert matrix_symmetry_diagnostics(matrix) == validate_local_response_symmetry(response)


def test_compare_local_responses_uses_new_interface_and_remains_not_casimir_ready():
    points, weights, amp = _inputs()
    rows = compare_local_responses_imag_axis(
        ("normal", "spm"),
        np.array([0.03]),
        points,
        temperature_K=20.0,
        pairing_params=amp,
        k_weights=weights,
    )

    assert [row["kind"] for row in rows] == ["normal", "spm"]
    assert all(row["valid_for_casimir_input"] is False for row in rows)
    assert rows[0]["source"] == "kubo_conductivity_imag_axis_from_model"
    assert rows[1]["source"] == "bdg_local_superconducting_response_imag_axis"
