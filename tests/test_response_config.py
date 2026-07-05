import numpy as np

from lno327.response.config import KuboConfig


def test_kubo_config_fields_and_defaults_are_stable():
    new = KuboConfig(omega_eV=0.12, temperature_eV=0.03)

    assert new.omega_eV == 0.12
    assert new.temperature_eV == 0.03
    assert new.fermi_level_eV == 0.0
    assert new.eta_eV == 1e-6
    assert new.output_si is True


def test_kubo_config_from_kelvin_uses_boltzmann_ev_per_kelvin():
    new = KuboConfig.from_kelvin(
        omega_eV=0.12,
        temperature_K=20.0,
        fermi_level_eV=0.01,
        eta_eV=2e-5,
        output_si=False,
    )

    assert new.omega_eV == 0.12
    np.testing.assert_allclose(new.temperature_eV, 20.0 * 8.617333262145e-5)
    assert new.fermi_level_eV == 0.01
    assert new.eta_eV == 2e-5
    assert new.output_si is False
