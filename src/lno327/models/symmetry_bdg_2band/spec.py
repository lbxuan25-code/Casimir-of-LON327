"""Unified spec for the symmetry-focused two-band BdG model."""

from __future__ import annotations

import numpy as np

from lno327.models.base import ModelChannel, ModelMetadata
from lno327.models.symmetry_bdg_2band.batched import (
    bdg_hamiltonian_from_pairing_batch as model_bdg_hamiltonian_from_pairing_batch,
    hopping_arrays,
    normal_hamiltonian_batch as model_normal_hamiltonian_batch,
    peierls_hamiltonian_contact_vertices_batch as model_peierls_hamiltonian_contact_vertices_batch,
    peierls_hamiltonian_vector_vertices_batch as model_peierls_hamiltonian_vector_vertices_batch,
    peierls_hamiltonian_vertices_batch as model_peierls_hamiltonian_vertices_batch,
)
from lno327.models.symmetry_bdg_2band.bdg import (
    bdg_hamiltonian as channel_bdg_hamiltonian,
)
from lno327.models.symmetry_bdg_2band.normal import (
    normal_hamiltonian as model_normal_hamiltonian,
)
from lno327.models.symmetry_bdg_2band.pairing import (
    pairing_matrix as channel_pairing_matrix,
)
from lno327.models.symmetry_bdg_2band.parameters import (
    BASIS,
    PairingChannel,
    TwoBandParameters,
)
from lno327.models.symmetry_bdg_2band.peierls import (
    normal_state_hamiltonian_from_hoppings as peierls_normal_state_hamiltonian_from_hoppings,
)
from lno327.models.symmetry_bdg_2band.peierls import (
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex as model_peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex as model_peierls_hamiltonian_vector_vertex,
    peierls_vertex_ward_residual as model_peierls_vertex_ward_residual,
)
from lno327.models.symmetry_bdg_2band.vertices import (
    mass_operator as model_mass_operator,
    velocity_operator as model_velocity_operator,
)


class SymmetryBdG2BandSpec:
    def __init__(self, params: TwoBandParameters | None = None) -> None:
        self.params = params or TwoBandParameters()
        self._hopping_terms = tuple(normal_state_hopping_terms(self.params))
        vectors, matrices = hopping_arrays(self._hopping_terms)
        vectors.setflags(write=False)
        matrices.setflags(write=False)
        self._hopping_vectors = vectors
        self._hopping_matrices = matrices

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="symmetry_bdg_2band",
            basis=BASIS,
            description="Symmetry-focused two-band BdG model",
        )

    def channels(self) -> tuple[ModelChannel, ...]:
        return (
            ModelChannel("normal"),
            ModelChannel("spm"),
            ModelChannel("dwave"),
        )

    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        return model_normal_hamiltonian(kx, ky, self.params)

    def normal_hamiltonian_batch(self, k_points: np.ndarray) -> np.ndarray:
        """Return normal-state Hamiltonians with shape (..., 2, 2)."""

        return model_normal_hamiltonian_batch(k_points, self.params)

    def pairing_matrix(
        self,
        kx: float,
        ky: float,
        channel: PairingChannel,
    ) -> np.ndarray:
        return channel_pairing_matrix(channel, kx, ky, self.params)

    def bdg_hamiltonian(
        self,
        kx: float,
        ky: float,
        channel: PairingChannel,
    ) -> np.ndarray:
        return channel_bdg_hamiltonian(kx, ky, channel, self.params)

    def bdg_hamiltonian_from_pairing_batch(
        self,
        k_points: np.ndarray,
        pairing: np.ndarray,
    ) -> np.ndarray:
        """Return batched BdG Hamiltonians for explicit pairing matrices."""

        return model_bdg_hamiltonian_from_pairing_batch(
            k_points,
            pairing,
            self.params,
        )

    def velocity_operator(
        self,
        kx: float,
        ky: float,
        direction: str,
    ) -> np.ndarray:
        return model_velocity_operator(kx, ky, direction, self.params)

    def mass_operator(
        self,
        kx: float,
        ky: float,
        i: str,
        j: str,
    ) -> np.ndarray:
        return model_mass_operator(kx, ky, i, j, self.params)

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
            self.params,
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
            self.params,
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
            self.params,
            terms,
        )

    def peierls_hamiltonian_vector_vertices_batch(
        self,
        k_points: np.ndarray,
        q_model: np.ndarray,
    ) -> np.ndarray:
        """Return both normal-state Peierls vector vertices in one batch."""

        return model_peierls_hamiltonian_vector_vertices_batch(
            k_points,
            q_model,
            self._hopping_vectors,
            self._hopping_matrices,
        )

    def peierls_hamiltonian_contact_vertices_batch(
        self,
        k_points: np.ndarray,
        q_model: np.ndarray,
    ) -> np.ndarray:
        """Return all normal-state Peierls contact vertices in one batch."""

        return model_peierls_hamiltonian_contact_vertices_batch(
            k_points,
            q_model,
            self._hopping_vectors,
            self._hopping_matrices,
        )

    def peierls_hamiltonian_vertices_batch(
        self,
        k_points: np.ndarray,
        q_model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return vector/contact batches while sharing hopping phases."""

        return model_peierls_hamiltonian_vertices_batch(
            k_points,
            q_model,
            self._hopping_vectors,
            self._hopping_matrices,
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
            self.params,
            terms,
        )
