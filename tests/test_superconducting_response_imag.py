from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from lno327 import (
    KuboConfig,
    PairingAmplitudes,
    bdg_superconducting_response_imag_axis,
    k_weights,
    uniform_bz_mesh,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_superconducting_response_imag.py"
SPEC = spec_from_file_location("diagnose_superconducting_response_imag", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnose = module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def test_superconducting_response_rejects_zero_omega():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.0, temperature_K=30.0, eta_eV=0.02, output_si=False)

    with pytest.raises(ValueError, match="omega_eV must be positive"):
        bdg_superconducting_response_imag_axis(
            mesh,
            config,
            "spm",
            PairingAmplitudes(delta0_eV=0.04),
            weights,
        )


def test_superconducting_response_scan_rejects_matsubara_zero():
    with pytest.raises(ValueError, match="1 <= min <= max"):
        diagnose.scan_kind(
            kind="spm",
            delta0_eV=0.04,
            nk=4,
            temperature_K=30.0,
            matsubara_min=0,
            matsubara_max=1,
            eta_eV=1e-4,
        )


def test_sigma_sc_equals_total_kernel_divided_by_omega():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.1, temperature_K=30.0, eta_eV=0.02, output_si=False)

    response = bdg_superconducting_response_imag_axis(
        mesh,
        config,
        "dwave",
        PairingAmplitudes(delta0_eV=0.04),
        weights,
    )

    np.testing.assert_allclose(response.sigma_like_response, response.total / config.omega_eV)


def test_superconducting_response_is_c4_symmetric_without_magnetic_field():
    mesh = uniform_bz_mesh(8)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.2, temperature_K=30.0, eta_eV=0.02, output_si=False)

    for kind in ("spm", "dwave"):
        sigma = bdg_superconducting_response_imag_axis(
            mesh,
            config,
            kind,
            PairingAmplitudes(delta0_eV=0.04),
            weights,
        ).sigma_like_response

        assert np.isclose(sigma[0, 0], sigma[1, 1], atol=1e-10)
        assert abs(sigma[0, 1]) < 1e-10
        assert abs(sigma[1, 0]) < 1e-10


def test_superconducting_response_npz_fields_are_complete(tmp_path):
    data = diagnose.scan_kind(
        kind="spm",
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_min=1,
        matsubara_max=1,
        eta_eV=1e-4,
    )
    npz_path, *_ = diagnose.save_outputs(data, tmp_path / "Sigma_SC")

    with np.load(npz_path) as loaded:
        assert set(loaded.files) == diagnose.REQUIRED_NPZ_FIELDS
