import numpy as np

from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.response.static_policy import local_response_matsubara_index


def test_static_policy_active_module_smoke():
    result = local_response_matsubara_index(
        "spm",
        0,
        20.0,
        policy="use_static_kernel",
        nk=2,
        pairing_params=PairingAmplitudes(delta0_eV=0.04),
    )

    assert result.status == "static_kernel"
    assert result.matrix is not None
    assert np.isfinite(result.matrix).all()
    assert result.valid_for_casimir_input is False
