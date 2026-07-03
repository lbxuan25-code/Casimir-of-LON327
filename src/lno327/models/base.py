"""Common model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    basis: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ModelChannel:
    name: str
    description: str = ""


class BdGModelSpec(Protocol):
    def metadata(self) -> ModelMetadata:
        ...

    def channels(self) -> tuple[ModelChannel, ...]:
        ...

    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        ...

    def pairing_matrix(self, kx: float, ky: float, channel: str) -> np.ndarray:
        ...

    def bdg_hamiltonian(self, kx: float, ky: float, channel: str) -> np.ndarray:
        ...

    def velocity_operator(self, kx: float, ky: float, direction: str) -> np.ndarray:
        ...

    def mass_operator(self, kx: float, ky: float, i: str, j: str) -> np.ndarray:
        ...


class PeierlsFiniteQModelSpec(Protocol):
    def hopping_terms(self):
        ...

    def normal_hamiltonian_from_hoppings(
        self,
        kx: float,
        ky: float,
        hopping_terms=None,
    ) -> np.ndarray:
        ...

    def peierls_hamiltonian_vector_vertex(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        direction: str,
        hopping_terms=None,
    ) -> np.ndarray:
        ...

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
        ...
