"""Unified spec for the symmetry-focused two-band BdG model."""

from __future__ import annotations

import numpy as np

from lno327.models.base import ModelChannel, ModelMetadata
from lno327.models.symmetry_bdg_2band.bdg import bdg_hamiltonian as channel_bdg_hamiltonian
from lno327.models.symmetry_bdg_2band.normal import normal_hamiltonian as model_normal_hamiltonian
from lno327.models.symmetry_bdg_2band.pairing import pairing_matrix as channel_pairing_matrix
from lno327.models.symmetry_bdg_2band.parameters import BASIS, PairingChannel, TwoBandParameters
from lno327.models.symmetry_bdg_2band.vertices import (
    mass_operator as model_mass_operator,
    velocity_operator as model_velocity_operator,
)


class SymmetryBdG2BandSpec:
    def __init__(self, params: TwoBandParameters | None = None) -> None:
        self.params = params or TwoBandParameters()

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="symmetry_bdg_2band",
            basis=BASIS,
            description="Symmetry-focused two-band BdG model",
        )

    def channels(self) -> tuple[ModelChannel, ...]:
        return (
            ModelChannel("normal"),
            ModelChannel("spp"),
            ModelChannel("spm"),
            ModelChannel("dwave"),
        )

    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        return model_normal_hamiltonian(kx, ky, self.params)

    def pairing_matrix(self, kx: float, ky: float, channel: PairingChannel) -> np.ndarray:
        return channel_pairing_matrix(channel, kx, ky, self.params)

    def bdg_hamiltonian(self, kx: float, ky: float, channel: PairingChannel) -> np.ndarray:
        return channel_bdg_hamiltonian(kx, ky, channel, self.params)

    def velocity_operator(self, kx: float, ky: float, direction: str) -> np.ndarray:
        return model_velocity_operator(kx, ky, direction, self.params)

    def mass_operator(self, kx: float, ky: float, i: str, j: str) -> np.ndarray:
        return model_mass_operator(kx, ky, i, j, self.params)
