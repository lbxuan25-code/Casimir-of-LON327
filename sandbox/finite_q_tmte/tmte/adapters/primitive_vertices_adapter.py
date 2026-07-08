"""Primitive EM vertex adapter for target-basis construction."""

from __future__ import annotations

import numpy as np

from lno327.bdg.finite_q import density_vertex
from lno327.response.finite_q_bdg import bdg_contact_vertex_from_spec, bdg_vector_vertex_from_spec


def primitive_source_vertices(spec: object, kx: float, ky: float, qx: float, qy: float, *, current_vertex: str = "peierls") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-side primitive vertices Gamma0,Gammax,Gammay."""

    dim = np.asarray(spec.normal_hamiltonian(kx, ky)).shape[0]
    gamma0 = density_vertex(int(dim))
    gammax = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", current_vertex)
    gammay = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", current_vertex)
    return gamma0, gammax, gammay


def primitive_observable_vertices(spec: object, kx: float, ky: float, qx: float, qy: float, *, current_vertex: str = "peierls") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-side primitive vertices using the existing observable sign convention."""

    gamma0, gammax, gammay = primitive_source_vertices(spec, kx, ky, qx, qy, current_vertex=current_vertex)
    return gamma0, -gammax, -gammay


def primitive_spatial_contact_vertices(spec: object, kx: float, ky: float, qx: float, qy: float, *, current_vertex: str = "peierls") -> dict[tuple[str, str], np.ndarray]:
    """Return primitive spatial contact vertices M_ij."""

    return {
        (di, dj): bdg_contact_vertex_from_spec(spec, kx, ky, qx, qy, di, dj, current_vertex)
        for di in ("x", "y")
        for dj in ("x", "y")
    }

