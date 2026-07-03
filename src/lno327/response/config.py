"""Configuration containers for local response calculations."""

from __future__ import annotations

from dataclasses import dataclass

from lno327.constants import KB_EV_PER_K


@dataclass(frozen=True)
class KuboConfig:
    """Inputs for band-basis Kubo response calculations."""

    omega_eV: float
    temperature_eV: float
    fermi_level_eV: float = 0.0
    eta_eV: float = 1e-6
    output_si: bool = True

    @classmethod
    def from_kelvin(
        cls,
        omega_eV: float,
        temperature_K: float,
        fermi_level_eV: float = 0.0,
        eta_eV: float = 1e-6,
        output_si: bool = True,
    ) -> "KuboConfig":
        return cls(
            omega_eV=omega_eV,
            temperature_eV=temperature_K * KB_EV_PER_K,
            fermi_level_eV=fermi_level_eV,
            eta_eV=eta_eV,
            output_si=output_si,
        )
