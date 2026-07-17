"""Outer in-plane momentum quadrature for Casimir-Lifshitz integration.

The microscopic response uses dimensionless model momenta while the Lifshitz
measure is an SI in-plane wavevector integral.  This module fixes that boundary
without evaluating any material response.

The radial coordinate is

    u = 2 Q d,

so that ``Q = u / (2 d)`` and

    d^2 Q / (2 pi)^2 = u du dphi / (16 pi^2 d^2).

The full angular interval ``[0, 2 pi)`` is always retained.  No lattice or plate
symmetry reduction is assumed by this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.constants import KB


_TWO_PI = 2.0 * np.pi


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape (N, 2)")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OuterQPolarGrid:
    """Tensor-product polar quadrature for ``integral d^2Q/(2pi)^2``."""

    u: np.ndarray
    phi_rad: np.ndarray
    q_si_m_inv: np.ndarray
    q_model: np.ndarray
    measure_weights_m_inv2: np.ndarray
    separation_m: float
    lattice_a_x_m: float
    lattice_a_y_m: float
    u_max: float
    radial_order: int
    angular_order: int
    angular_offset_fraction: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", _readonly_vector(self.u, "u"))
        object.__setattr__(self, "phi_rad", _readonly_vector(self.phi_rad, "phi_rad"))
        object.__setattr__(
            self,
            "q_si_m_inv",
            _readonly_points(self.q_si_m_inv, "q_si_m_inv"),
        )
        object.__setattr__(self, "q_model", _readonly_points(self.q_model, "q_model"))
        object.__setattr__(
            self,
            "measure_weights_m_inv2",
            _readonly_vector(self.measure_weights_m_inv2, "measure_weights_m_inv2"),
        )
        count = len(self.u)
        if not (
            len(self.phi_rad) == count
            and len(self.q_si_m_inv) == count
            and len(self.q_model) == count
            and len(self.measure_weights_m_inv2) == count
        ):
            raise ValueError("all outer-q grid arrays must have the same node count")
        if count != int(self.radial_order) * int(self.angular_order):
            raise ValueError("node count must equal radial_order * angular_order")
        if not np.all(self.measure_weights_m_inv2 > 0.0):
            raise ValueError("outer-q measure weights must be positive")
        for name in ("separation_m", "lattice_a_x_m", "lattice_a_y_m", "u_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("radial_order", "angular_order"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        offset = float(self.angular_offset_fraction)
        if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
            raise ValueError("angular_offset_fraction must lie in [0, 1)")
        object.__setattr__(self, "angular_offset_fraction", offset)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def node_count(self) -> int:
        return len(self.u)

    @property
    def q_max_m_inv(self) -> float:
        return float(self.u_max / (2.0 * self.separation_m))

    @property
    def disk_measure_m_inv2(self) -> float:
        """Exact ``integral_{|Q|<=Qmax} d^2Q/(2pi)^2``."""

        return float(self.q_max_m_inv**2 / (4.0 * np.pi))


def build_outer_q_polar_grid(
    *,
    separation_m: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
    u_max: float,
    radial_order: int,
    angular_order: int,
    angular_offset_fraction: float = 0.5,
) -> OuterQPolarGrid:
    """Build the fixed full-angle outer-q quadrature.

    Gauss-Legendre quadrature is used on ``u in [0, u_max]``.  The angular rule
    is the periodic equal-weight trapezoidal rule with an optional fractional
    cell offset.  The default half-cell offset avoids privileging crystal axes;
    offset zero remains available as an independent cut/phase audit.
    """

    d = float(separation_m)
    ax = float(lattice_a_x_m)
    ay = float(lattice_a_y_m)
    upper = float(u_max)
    nr = int(radial_order)
    nphi = int(angular_order)
    offset = float(angular_offset_fraction)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("separation_m must be finite and positive")
    if not np.isfinite(ax) or ax <= 0.0 or not np.isfinite(ay) or ay <= 0.0:
        raise ValueError("lattice constants must be finite and positive")
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("u_max must be finite and positive")
    if nr <= 0 or nphi <= 0:
        raise ValueError("radial_order and angular_order must be positive")
    if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
        raise ValueError("angular_offset_fraction must lie in [0, 1)")

    roots, root_weights = np.polynomial.legendre.leggauss(nr)
    radial_u = 0.5 * upper * (roots + 1.0)
    radial_du_weights = 0.5 * upper * root_weights
    angular_phi = _TWO_PI * (np.arange(nphi, dtype=float) + offset) / float(nphi)
    angular_weight = _TWO_PI / float(nphi)

    u_mesh, phi_mesh = np.meshgrid(radial_u, angular_phi, indexing="ij")
    radial_weight_mesh, _ = np.meshgrid(
        radial_du_weights,
        angular_phi,
        indexing="ij",
    )
    q_radius = u_mesh / (2.0 * d)
    qx = q_radius * np.cos(phi_mesh)
    qy = q_radius * np.sin(phi_mesh)
    q_si = np.column_stack([qx.ravel(), qy.ravel()])
    q_model = np.column_stack([(ax * qx).ravel(), (ay * qy).ravel()])
    weights = (
        u_mesh
        * radial_weight_mesh
        * angular_weight
        / (16.0 * np.pi**2 * d**2)
    ).ravel()

    exact_measure = upper**2 / (16.0 * np.pi * d**2)
    weight_sum = float(np.sum(weights))
    weight_error = abs(weight_sum - exact_measure)
    tolerance = 64.0 * np.finfo(float).eps * max(exact_measure, 1.0)
    if weight_error > tolerance:
        raise RuntimeError("outer-q quadrature weights fail the exact disk-measure check")

    return OuterQPolarGrid(
        u=u_mesh.ravel(),
        phi_rad=phi_mesh.ravel(),
        q_si_m_inv=q_si,
        q_model=q_model,
        measure_weights_m_inv2=weights,
        separation_m=d,
        lattice_a_x_m=ax,
        lattice_a_y_m=ay,
        u_max=upper,
        radial_order=nr,
        angular_order=nphi,
        angular_offset_fraction=offset,
        metadata={
            "schema": "outer-q-polar-grid-v1",
            "radial_variable": "u = 2 Q d",
            "measure_formula": "d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2)",
            "model_momentum_formula": "q_model = (a_x Q_x, a_y Q_y)",
            "radial_rule": "Gauss-Legendre on finite [0, u_max]",
            "angular_rule": "full-period equal-weight trapezoidal",
            "angular_symmetry_reduction": False,
            "q_zero_node_present": False,
            "q_max_m_inv": upper / (2.0 * d),
            "weight_sum_m_inv2": weight_sum,
            "exact_disk_measure_m_inv2": exact_measure,
            "absolute_weight_sum_error_m_inv2": weight_error,
            "max_abs_q_model_x": float(np.max(np.abs(q_model[:, 0]))),
            "max_abs_q_model_y": float(np.max(np.abs(q_model[:, 1]))),
        },
    )


def integrate_outer_q(values: np.ndarray, grid: OuterQPolarGrid) -> np.ndarray | float:
    """Integrate node values over ``d^2Q/(2pi)^2``.

    The final axis of ``values`` must be the outer-q node axis.  Leading axes are
    preserved, allowing a complete Matsubara batch to be reduced at once.
    """

    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != grid.node_count:
        raise ValueError("the final values axis must match the outer-q node count")
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError("outer-q values must be finite")
    result = np.tensordot(array, grid.measure_weights_m_inv2, axes=([-1], [0]))
    if np.ndim(result) == 0:
        return result.item()
    return result


def matsubara_prime_weights(indices: Sequence[int]) -> np.ndarray:
    """Return the standard prime-sum weights, with one half at ``n=0``."""

    values = np.asarray(tuple(int(index) for index in indices), dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Matsubara indices must be a nonempty one-dimensional sequence")
    if np.any(values < 0) or len(set(values.tolist())) != len(values):
        raise ValueError("Matsubara indices must be unique and non-negative")
    weights = np.ones(values.shape, dtype=float)
    weights[values == 0] = 0.5
    return weights


@dataclass(frozen=True)
class MatsubaraFreeEnergyPerArea:
    """Finite Matsubara partial sum of outer-q integrated logdet values."""

    total_J_m2: float
    contributions_J_m2: np.ndarray
    outer_q_integrals_m_inv2: np.ndarray
    prime_weights: np.ndarray
    matsubara_indices: np.ndarray
    temperature_K: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        total = float(self.total_J_m2)
        temperature = float(self.temperature_K)
        if not np.isfinite(total):
            raise ValueError("total_J_m2 must be finite")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        object.__setattr__(self, "total_J_m2", total)
        object.__setattr__(self, "temperature_K", temperature)
        for name in (
            "contributions_J_m2",
            "outer_q_integrals_m_inv2",
            "prime_weights",
            "matsubara_indices",
        ):
            value = np.array(getattr(self, name), copy=True)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        size = len(self.matsubara_indices)
        if not all(
            len(getattr(self, name)) == size
            for name in (
                "contributions_J_m2",
                "outer_q_integrals_m_inv2",
                "prime_weights",
            )
        ):
            raise ValueError("Matsubara result arrays must have matching lengths")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def free_energy_per_area_from_logdet(
    logdet_by_n_and_node: np.ndarray,
    *,
    matsubara_indices: Sequence[int],
    temperature_K: float,
    grid: OuterQPolarGrid,
) -> MatsubaraFreeEnergyPerArea:
    """Return a finite Matsubara partial sum in joules per square metre."""

    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    indices = np.asarray(tuple(int(index) for index in matsubara_indices), dtype=int)
    values = np.asarray(logdet_by_n_and_node, dtype=float)
    if values.shape != (len(indices), grid.node_count):
        raise ValueError("logdet array must have shape (n_indices, outer_q_nodes)")
    if not np.isfinite(values).all():
        raise ValueError("logdet values must be finite")
    prime = matsubara_prime_weights(indices)
    outer = np.asarray(integrate_outer_q(values, grid), dtype=float)
    contributions = KB * temperature * prime * outer
    return MatsubaraFreeEnergyPerArea(
        total_J_m2=float(np.sum(contributions)),
        contributions_J_m2=contributions,
        outer_q_integrals_m_inv2=outer,
        prime_weights=prime,
        matsubara_indices=indices,
        temperature_K=temperature,
        metadata={
            "schema": "matsubara-free-energy-per-area-v1",
            "formula": "F/A = k_B T sum_n' integral d^2Q/(2pi)^2 logdet",
            "zero_matsubara_prime_weight": 0.5,
            "tail_included": False,
            "partial_sum_only": True,
        },
    )
