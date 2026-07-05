"""Model adapters for finite-q validation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz as build_four_orbital_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.collective import (
    SymmetryTwoBandPairingAmplitudes,
    build_pairing_ansatz as build_two_band_ansatz,
)
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


@dataclass(frozen=True)
class FiniteQValidationModel:
    name: str
    spec: object
    pairing_names: tuple[str, ...]
    default_pairings: tuple[str, ...]
    default_delta0_eV: float
    primary_validation_model: bool
    _build_ansatz: Callable[[str, str], object]
    _build_pairing_params: Callable[[float], object]

    def build_ansatz(self, pairing_name: str, phase_vertex: str = "bond_endpoint_gauge"):
        self.require_pairing(pairing_name)
        return self._build_ansatz(pairing_name, phase_vertex)

    def build_pairing_params(self, delta0_eV: float | None = None):
        return self._build_pairing_params(self.default_delta0_eV if delta0_eV is None else float(delta0_eV))

    def require_pairing(self, pairing_name: str) -> None:
        if pairing_name not in self.pairing_names:
            choices = ", ".join(self.pairing_names)
            raise ValueError(f"pairing {pairing_name!r} is not supported by {self.name}; choices: {choices}")

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.name,
            "pairing_names": list(self.pairing_names),
            "default_pairings": list(self.default_pairings),
            "default_delta0_eV": float(self.default_delta0_eV),
            "primary_validation_model": bool(self.primary_validation_model),
            "valid_for_casimir_input": False,
            "spec_metadata": self.spec.metadata().__dict__,
        }


def _two_band_model() -> FiniteQValidationModel:
    default_delta = 0.1
    params = TwoBandParameters(delta_s=default_delta, delta_d=default_delta)
    return FiniteQValidationModel(
        name="symmetry_bdg_2band",
        spec=SymmetryBdG2BandSpec(params),
        pairing_names=("normal", "spm", "dwave"),
        default_pairings=("spm", "dwave"),
        default_delta0_eV=default_delta,
        primary_validation_model=True,
        _build_ansatz=lambda pairing, phase_vertex: build_two_band_ansatz(pairing, phase_vertex=phase_vertex),
        _build_pairing_params=lambda delta: SymmetryTwoBandPairingAmplitudes(delta0_eV=delta, delta_s=delta, delta_d=delta),
    )


def _four_orbital_model() -> FiniteQValidationModel:
    default_delta = 0.04
    amp = PairingAmplitudes(delta0_eV=default_delta)
    return FiniteQValidationModel(
        name="lno327_four_orbital",
        spec=LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        pairing_names=("onsite_s", "spm", "dwave"),
        default_pairings=("onsite_s", "spm", "dwave"),
        default_delta0_eV=default_delta,
        primary_validation_model=False,
        _build_ansatz=lambda pairing, phase_vertex: build_four_orbital_ansatz(pairing, phase_vertex=phase_vertex),
        _build_pairing_params=lambda delta: PairingAmplitudes(delta0_eV=delta),
    )


def available_finite_q_validation_models() -> tuple[str, ...]:
    return ("symmetry_bdg_2band", "lno327_four_orbital")


def get_finite_q_validation_model(name: str) -> FiniteQValidationModel:
    if name == "symmetry_bdg_2band":
        return _two_band_model()
    if name == "lno327_four_orbital":
        return _four_orbital_model()
    choices = ", ".join(available_finite_q_validation_models())
    raise ValueError(f"unknown finite-q validation model {name!r}; choices: {choices}")
