from pathlib import Path

import numpy as np

from lno327 import api
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes


ROOT = Path(__file__).resolve().parents[1]


def test_api_no_longer_imports_old_active_response_modules():
    text = (ROOT / "src/lno327/api.py").read_text(encoding="utf-8")
    forbidden = (
        "from .bdg_response",
        "from lno327.bdg_response",
        "from .conductivity",
        "from lno327.conductivity",
        "from .response_interface",
        "from lno327.response_interface",
        "from .static_response",
        "from lno327.static_response",
    )
    for needle in forbidden:
        assert needle not in text


def test_new_response_modules_do_not_import_old_active_modules():
    checks = {
        "src/lno327/response/local_interface.py": (
            "from .bdg_response",
            "from lno327.bdg_response",
            "from .conductivity",
            "from lno327.conductivity",
        ),
        "src/lno327/response/static_policy.py": (
            "from .bdg_response",
            "from lno327.bdg_response",
            "from .conductivity",
            "from lno327.conductivity",
            "from .response_interface",
            "from lno327.response_interface",
        ),
    }
    for relative, forbidden in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text


def test_api_convenience_names_still_call_new_model_driven_response():
    points = api.uniform_bz_mesh(2)
    weights = np.full(points.shape[0], 1.0 / points.shape[0])
    config = api.KuboConfig.from_kelvin(omega_eV=0.03, temperature_K=20.0, eta_eV=1e-4, output_si=False)

    normal = api.kubo_conductivity_imag_axis(points, config, weights)
    bdg_response = api.bdg_superconducting_response_imag_axis(
        points,
        config,
        "spm",
        PairingAmplitudes(delta0_eV=0.04),
        weights,
    )
    kernel = api.bdg_total_kernel_imag_axis(
        points,
        config,
        "spm",
        PairingAmplitudes(delta0_eV=0.04),
        weights,
    )

    assert normal.matrix().shape == (2, 2)
    assert bdg_response.sigma_like_response.shape == (2, 2)
    assert kernel.total.shape == (2, 2)
    assert api.local_response_matsubara_index("spm", 0, 20.0, policy="skip", nk=2).valid_for_casimir_input is False
