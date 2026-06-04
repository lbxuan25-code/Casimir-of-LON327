from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from lno327 import (
    KuboConfig,
    PairingAmplitudes,
    bdg_diamagnetic_kernel,
    bdg_paramagnetic_kernel_imag_axis,
    bdg_total_kernel_imag_axis,
    k_weights,
    uniform_bz_mesh,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bdg" / "diagnose_bdg_total_kernel_imag.py"
SPEC = spec_from_file_location("diagnose_bdg_total_kernel_imag", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnose = module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def test_bdg_total_kernel_imag_axis_returns_complex_2x2_matrix():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.1, temperature_K=30.0, eta_eV=0.02, output_si=False)

    components = bdg_total_kernel_imag_axis(mesh, config, "spm", PairingAmplitudes(delta0_eV=0.04), weights)

    assert components.total.shape == (2, 2)
    assert components.total.dtype == complex
    assert np.isfinite(components.total).all()


def test_bdg_total_kernel_equals_diamagnetic_minus_paramagnetic():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.1, temperature_K=30.0, eta_eV=0.02, output_si=False)
    params = PairingAmplitudes(delta0_eV=0.04)

    components = bdg_total_kernel_imag_axis(mesh, config, "dwave", params, weights)
    para = bdg_paramagnetic_kernel_imag_axis(mesh, config, "dwave", params, weights)
    dia = bdg_diamagnetic_kernel("dwave", params, mesh, config, weights)

    np.testing.assert_allclose(components.paramagnetic, para)
    np.testing.assert_allclose(components.diamagnetic, dia)
    np.testing.assert_allclose(components.total, dia - para)


def test_bdg_total_kernel_is_c4_symmetric_without_magnetic_field():
    mesh = uniform_bz_mesh(8)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.2, temperature_K=30.0, eta_eV=0.02, output_si=False)

    for kind in ("spm", "dwave"):
        total = bdg_total_kernel_imag_axis(
            mesh,
            config,
            kind,
            PairingAmplitudes(delta0_eV=0.04),
            weights,
        ).total

        assert np.isclose(total[0, 0], total[1, 1], atol=1e-10)
        assert abs(total[0, 1]) < 1e-10
        assert abs(total[1, 0]) < 1e-10


def test_total_kernel_scan_fields_are_complete(tmp_path):
    data = diagnose.scan_kind(
        kind="spm",
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_min=1,
        matsubara_max=1,
        eta_eV=1e-4,
    )
    npz_path, *_ = diagnose.save_outputs(data, tmp_path / "total_kernel")

    with np.load(npz_path) as loaded:
        assert set(loaded.files) == diagnose.REQUIRED_NPZ_FIELDS
