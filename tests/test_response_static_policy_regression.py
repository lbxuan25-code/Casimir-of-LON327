import numpy as np
import pytest

from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.response.static_policy import (
    local_response_matsubara_index as new_local_response_matsubara_index,
    matsubara_response_series,
)


def _assert_static_common_fields_are_well_formed(new, kind, n, policy):
    assert new.kind == kind
    assert new.n == n
    assert new.omega_eV >= 0.0
    assert new.policy == policy
    assert isinstance(new.status, str)
    assert isinstance(new.approximate, bool)
    assert isinstance(new.unit_label, str)
    assert new.valid_for_casimir_input is False
    if new.matrix is not None:
        assert new.matrix.shape == (2, 2)
        assert np.all(np.isfinite(new.matrix))


@pytest.mark.parametrize(
    ("kind", "n", "policy"),
    (
        ("spm", 1, "skip"),
        ("spm", 0, "skip"),
        ("spm", 0, "extrapolate_from_lowest_matsubara"),
        ("normal", 0, "use_static_kernel"),
        ("spm", 0, "use_static_kernel"),
    ),
)
def test_new_static_policy_result_is_well_formed(kind, n, policy):
    amp = PairingAmplitudes(delta0_eV=0.04)
    kwargs = {
        "kind": kind,
        "n": n,
        "temperature_K": 20.0,
        "policy": policy,
        "nk": 2,
        "eta_eV": 1e-4,
        "pairing_params": amp,
    }

    new = new_local_response_matsubara_index(**kwargs)

    _assert_static_common_fields_are_well_formed(new, kind, n, policy)
    assert new.valid_for_casimir_input is False
    if n == 0:
        assert any("n=0" in note or "diagnostic" in note for note in new.notes)


def test_static_policy_rejects_bad_inputs_and_series_is_not_casimir_ready():
    with pytest.raises(ValueError, match="n must be non-negative"):
        new_local_response_matsubara_index("spm", -1, 20.0, nk=2)
    with pytest.raises(ValueError, match="Unknown static response policy"):
        new_local_response_matsubara_index("spm", 0, 20.0, policy="bad", nk=2)  # type: ignore[arg-type]

    series = matsubara_response_series(
        "spm",
        np.array([0, 1]),
        20.0,
        policy="extrapolate_from_lowest_matsubara",
        nk=2,
        pairing_params=PairingAmplitudes(delta0_eV=0.04),
    )
    assert [item.n for item in series] == [0, 1]
    assert all(item.valid_for_casimir_input is False for item in series)
