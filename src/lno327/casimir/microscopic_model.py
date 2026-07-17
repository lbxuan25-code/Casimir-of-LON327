"""Finite-q microscopic model adapter for the production Casimir chain.

The adapter exposes the active symmetry-based two-band BdG model. Point
certification and full Casimir readiness remain separate concerns; constructing
this adapter alone never authorizes a production result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lno327.models.symmetry_bdg_2band.collective import (
    SymmetryTwoBandPairingAmplitudes,
    build_pairing_ansatz,
)
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


@dataclass(frozen=True)
class FiniteQMicroscopicModel:
    """Production-owned adapter for one finite-q microscopic model family."""

    name: str
    spec: object
    pairing_names: tuple[str, ...]
    default_pairings: tuple[str, ...]
    default_delta0_eV: float
    primary_model: bool
    _build_ansatz: Callable[[str, str], object]
    _build_pairing_params: Callable[[float], object]

    def build_ansatz(
        self,
        pairing_name: str,
        phase_vertex: str = "bond_endpoint_gauge",
    ):
        self.require_pairing(pairing_name)
        return self._build_ansatz(pairing_name, phase_vertex)

    def build_pairing_params(self, delta0_eV: float | None = None):
        value = self.default_delta0_eV if delta0_eV is None else float(delta0_eV)
        return self._build_pairing_params(value)

    def require_pairing(self, pairing_name: str) -> None:
        if pairing_name not in self.pairing_names:
            choices = ", ".join(self.pairing_names)
            raise ValueError(
                f"pairing {pairing_name!r} is not supported by {self.name}; "
                f"choices: {choices}"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.name,
            "pairing_names": list(self.pairing_names),
            "default_pairings": list(self.default_pairings),
            "default_delta0_eV": float(self.default_delta0_eV),
            "primary_model": bool(self.primary_model),
            "valid_for_casimir_input": False,
            "casimir_readiness_reason": (
                "model construction does not replace finite-q point certification"
            ),
            "spec_metadata": self.spec.metadata().__dict__,
        }


def _two_band_model() -> FiniteQMicroscopicModel:
    default_delta = 0.1
    params = TwoBandParameters(delta_s=default_delta, delta_d=default_delta)
    return FiniteQMicroscopicModel(
        name="symmetry_bdg_2band",
        spec=SymmetryBdG2BandSpec(params),
        pairing_names=("normal", "spm", "dwave"),
        default_pairings=("spm", "dwave"),
        default_delta0_eV=default_delta,
        primary_model=True,
        _build_ansatz=lambda pairing, phase_vertex: build_pairing_ansatz(
            pairing,
            phase_vertex=phase_vertex,
        ),
        _build_pairing_params=lambda delta: SymmetryTwoBandPairingAmplitudes(
            delta0_eV=delta,
            delta_s=delta,
            delta_d=delta,
        ),
    )


def available_finite_q_microscopic_models() -> tuple[str, ...]:
    """Return the finite-q model families supported by the production chain."""

    return ("symmetry_bdg_2band",)


def get_finite_q_microscopic_model(name: str) -> FiniteQMicroscopicModel:
    """Return the requested production finite-q model adapter."""

    if name == "symmetry_bdg_2band":
        return _two_band_model()
    choices = ", ".join(available_finite_q_microscopic_models())
    raise ValueError(f"unknown finite-q microscopic model {name!r}; choices: {choices}")


__all__ = [
    "FiniteQMicroscopicModel",
    "available_finite_q_microscopic_models",
    "get_finite_q_microscopic_model",
]
