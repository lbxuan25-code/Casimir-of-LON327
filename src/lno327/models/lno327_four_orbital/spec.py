"""Unified spec for the LNO327 four-orbital BdG model."""

from __future__ import annotations

import numpy as np

from lno327.models.base import ModelChannel, ModelMetadata
from lno327.models.lno327_four_orbital.bdg import bdg_hamiltonian as assemble_bdg_hamiltonian
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian
from lno327.models.lno327_four_orbital.pairing import pairing_matrix as channel_pairing_matrix
from lno327.models.lno327_four_orbital.parameters import ORBITAL_BASIS, NormalStateParameters, PairingAmplitudes
from lno327.models.lno327_four_orbital.peierls import (
    normal_state_hamiltonian_from_hoppings as peierls_normal_state_hamiltonian_from_hoppings,
)
from lno327.models.lno327_four_orbital.peierls import (
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex as model_peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex as model_peierls_hamiltonian_vector_vertex,
    peierls_vertex_ward_residual as model_peierls_vertex_ward_residual,
)
from lno327.models.lno327_four_orbital.vertices import normal_state_mass_operator, normal_state_velocity_operator


class LNO327FourOrbitalSpec:
    def __init__(
        self,
        normal_params: NormalStateParameters | None = None,
        pairing_amplitudes: PairingAmplitudes | None = None,
    ) -> None:
        self.normal_params = normal_params or NormalStateParameters()
        self.pairing_amplitudes = pairing_amplitudes or PairingAmplitudes()
        self._hopping_terms = tuple(normal_state_hopping_terms(self.normal_params))

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="lno327_four_orbital",
            basis=ORBITAL_BASIS,
            description="LNO327 bilayer four-orbital BdG model",
        )

    def channels(self) -> tuple[ModelChannel, ...]:
        return (
            ModelChannel("normal"),
            ModelChannel("spm"),
            ModelChannel("dwave"),
        )

    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        return normal_state_hamiltonian(kx, ky, self.normal_params)

    def pairing_matrix(self, kx: float, ky: float, channel: str) -> np.ndarray:
        if channel == "normal":
            return np.zeros((4, 4), dtype=complex)
        if channel in {"spm", "dwave"}:
            return channel_pairing_matrix(channel, kx, ky, self.pairing_amplitudes)
        raise ValueError("channel must be 'normal', 'spm', or 'dwave'")

    def bdg_hamiltonian(self, kx: float, ky: float, channel: str) -> np.ndarray:
        return assemble_bdg_hamiltonian(
            kx,
            ky,
            self.pairing_matrix(kx, ky, channel),
            normal_state=self.normal_hamiltonian,
        )

    def velocity_operator(self, kx: float, ky: float, direction: str) -> np.ndarray:
        return normal_state_velocity_operator(kx, ky, direction, self.normal_params)

    def mass_operator(self, kx: float, ky: float, i: str, j: str) -> np.ndarray:
        return normal_state_mass_operator(kx, ky, i, j, self.normal_params)

    def hopping_terms(self):
        return self._hopping_terms

    def normal_hamiltonian_from_hoppings(
        self,
        kx: float,
        ky: float,
        hopping_terms=None,
    ) -> np.ndarray:
        terms = self._hopping_terms if hopping_terms is None else hopping_terms
        return peierls_normal_state_hamiltonian_from_hoppings(
            kx,
            ky,
            self.normal_params,
            terms,
        )

    def peierls_hamiltonian_vector_vertex(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        direction: str,
        hopping_terms=None,
    ) -> np.ndarray:
        terms = self._hopping_terms if hopping_terms is None else hopping_terms
        return model_peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            direction,
            self.normal_params,
            terms,
        )

    def peierls_hamiltonian_contact_vertex(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        direction_i: str,
        direction_j: str,
        hopping_terms=None,
    ) -> np.ndarray:
        terms = self._hopping_terms if hopping_terms is None else hopping_terms
        return model_peierls_hamiltonian_contact_vertex(
            kx,
            ky,
            qx,
            qy,
            direction_i,
            direction_j,
            self.normal_params,
            terms,
        )

    def peierls_vertex_ward_residual(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        hopping_terms=None,
    ) -> tuple[float, float, float, float]:
        terms = self._hopping_terms if hopping_terms is None else hopping_terms
        return model_peierls_vertex_ward_residual(
            kx,
            ky,
            qx,
            qy,
            self.normal_params,
            terms,
        )
