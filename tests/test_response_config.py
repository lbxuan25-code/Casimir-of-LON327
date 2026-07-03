from lno327.conductivity import KuboConfig as OldKuboConfig
from lno327.response.config import KuboConfig


def test_kubo_config_fields_and_defaults_match_legacy():
    old = OldKuboConfig(omega_eV=0.12, temperature_eV=0.03)
    new = KuboConfig(omega_eV=0.12, temperature_eV=0.03)

    assert new.omega_eV == old.omega_eV
    assert new.temperature_eV == old.temperature_eV
    assert new.fermi_level_eV == old.fermi_level_eV
    assert new.eta_eV == old.eta_eV
    assert new.output_si == old.output_si


def test_kubo_config_from_kelvin_matches_legacy():
    old = OldKuboConfig.from_kelvin(
        omega_eV=0.12,
        temperature_K=20.0,
        fermi_level_eV=0.01,
        eta_eV=2e-5,
        output_si=False,
    )
    new = KuboConfig.from_kelvin(
        omega_eV=0.12,
        temperature_K=20.0,
        fermi_level_eV=0.01,
        eta_eV=2e-5,
        output_si=False,
    )

    assert new == KuboConfig(
        omega_eV=old.omega_eV,
        temperature_eV=old.temperature_eV,
        fermi_level_eV=old.fermi_level_eV,
        eta_eV=old.eta_eV,
        output_si=old.output_si,
    )
