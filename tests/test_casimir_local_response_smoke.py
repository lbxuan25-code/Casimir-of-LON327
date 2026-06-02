from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from lno327 import CasimirSetup, casimir_torque_integrand
from lno327.casimir import matsubara_frequency
from lno327.constants import E2_OVER_HBAR, SIGMA0

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "smoke"
    / "smoke_casimir_local_response.py"
)
SPEC = spec_from_file_location("smoke_casimir_local_response", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke = module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_toy_isotropic_tensor_torque_is_near_zero():
    setup = CasimirSetup(temperature=30.0, distance=30e-9)
    xi = matsubara_frequency(1, setup.temperature)
    tensor = smoke.toy_isotropic_tensor()

    torque = casimir_torque_integrand(setup, xi, 1e6, 0.2, 0.7, tensor, tensor)

    assert abs(torque) < 1e-20


def test_toy_anisotropic_tensor_torque_is_nonzero():
    setup = CasimirSetup(temperature=30.0, distance=30e-9)
    xi = matsubara_frequency(1, setup.temperature)
    tensor = smoke.toy_anisotropic_tensor()

    torque = casimir_torque_integrand(setup, xi, 1e6, 0.2, 0.7, tensor, tensor)

    assert abs(torque) > 1e-12


def test_normal_spm_dwave_response_passes_to_casimir_integrands():
    data = smoke.scan_smoke(
        kinds=["normal", "spm", "dwave"],
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_index=1,
        eta_eV=1e-4,
        distance_m=30e-9,
        k_parallel_m_inv=1e6,
        phi_rad=0.2,
        theta_rad=0.7,
    )

    assert set(data["kind"]) == {"normal", "spm", "dwave"}
    assert np.isfinite(data["energy_integrand"]).all()
    assert np.isfinite(data["torque_integrand"]).all()
    assert np.isfinite(data["response_isotropic_diagnostic"]).all()
    np.testing.assert_allclose(data["sheet_conductivity_xx"], E2_OVER_HBAR * data["response_xx"])
    np.testing.assert_allclose(
        data["reflection_dimensionless_xx"],
        (E2_OVER_HBAR / SIGMA0) * data["response_xx"],
    )
    assert set(data["response_unit_stage"]) == {"model_response"}


def test_smoke_output_fields_are_complete(tmp_path):
    data = smoke.scan_smoke(
        kinds=["normal", "spm", "dwave"],
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_index=1,
        eta_eV=1e-4,
        distance_m=30e-9,
        k_parallel_m_inv=1e6,
        phi_rad=0.2,
        theta_rad=0.7,
    )
    npz_path, _ = smoke.save_outputs(data, tmp_path / "casimir_smoke")

    assert smoke.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(npz_path) as loaded:
        assert smoke.REQUIRED_NPZ_FIELDS.issubset(loaded.files)
