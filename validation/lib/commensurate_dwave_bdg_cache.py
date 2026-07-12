"""Index-addressable BdG cache for commensurate d-wave Ward audits.

For ``q = 2*pi*(mx,my)/nk``, endpoint momenta ``k +/- q/2`` lie on either the
same shifted tensor subgrid or one of its complementary half-step partners.  This
module diagonalizes every required subgrid once and serves midpoint/minus/plus
bands by exact integer index maps.  It changes neither the primitive integrand nor
the compensated periodic sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from lno327.bdg.finite_q import phase_phase_direct_vertex
from lno327.response.finite_q_bdg import (
    bdg_contact_vertex_from_spec,
    bdg_vector_vertex_from_spec,
)
from lno327.response.occupations import fermi_function
from validation.lib.commensurate_periodic import CommensuratePeriodicGrid
from validation.lib.dwave_iterated_adaptive import (
    DWaveStaticIntegrandContext,
    _COLLECTIVE_CHANNELS,
    _COMPLEX_WIDTH,
    _COUNTERTERM_SLICE,
    _DIRECT_SLICE,
    _EM_CHANNELS,
    _EM_OBSERVABLE_SIGNS,
    _PHASE_DIRECT_SLICE,
    _UNIFIED_SLICE,
    _WARD_CONTACT_SLICE,
    _WARD_DELTA_V_SLICE,
    _WARD_EQUAL_SLICE,
    _static_factor_matrix,
)
from validation.lib.dwave_iterated_adaptive_fast import (
    _band_transform_stack,
    _batched_eigensystems,
    _thermal_density,
    _thermal_expectation_from_density,
    _unified_contraction,
    _ward_equal_contraction,
)


def _origin_key(origin: tuple[float, float]) -> tuple[float, float]:
    return tuple(float(round(value % 1.0, 14)) for value in origin)


@dataclass(frozen=True)
class CachedSubgridBands:
    origin: tuple[float, float]
    energies: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    build_wall_seconds: float

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        states = np.asarray(self.states, dtype=complex)
        occupations = np.asarray(self.occupations, dtype=float)
        if energies.ndim != 2:
            raise ValueError("cached energies must have shape (nk2, nb)")
        count, nb = energies.shape
        if states.shape != (count, nb, nb):
            raise ValueError("cached states must have shape (nk2, nb, nb)")
        if occupations.shape != (count, nb):
            raise ValueError("cached occupations must have shape (nk2, nb)")
        energies.setflags(write=False)
        states.setflags(write=False)
        occupations.setflags(write=False)
        object.__setattr__(self, "origin", _origin_key(self.origin))
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "build_wall_seconds", float(self.build_wall_seconds))


@dataclass(frozen=True)
class CommensurateDWaveBdGCache:
    context: DWaveStaticIntegrandContext
    nk: int
    mx: int
    my: int
    subgrids: Mapping[tuple[float, float], CachedSubgridBands]
    chunk_size: int
    build_wall_seconds: float

    @property
    def step(self) -> float:
        return 2.0 * np.pi / float(self.nk)

    @property
    def eigensystem_count(self) -> int:
        return len(self.subgrids) * self.nk * self.nk

    def subgrid(self, origin: tuple[float, float]) -> CachedSubgridBands:
        key = _origin_key(origin)
        try:
            return self.subgrids[key]
        except KeyError as exc:
            raise KeyError(f"subgrid origin {key} was not cached") from exc

    def endpoint_origin(self, origin: tuple[float, float]) -> tuple[float, float]:
        sx, sy = _origin_key(origin)
        return _origin_key(
            (
                sx + 0.5 * float(self.mx % 2),
                sy + 0.5 * float(self.my % 2),
            )
        )

    def source_indices(
        self,
        points: np.ndarray,
        origin: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        sx, sy = _origin_key(origin)
        values = np.asarray(points, dtype=float)
        ix_float = (values[:, 0] + np.pi) / self.step - sx
        iy_float = (values[:, 1] + np.pi) / self.step - sy
        ix = np.rint(ix_float).astype(np.int64) % self.nk
        iy = np.rint(iy_float).astype(np.int64) % self.nk
        reconstructed = np.column_stack(
            (
                -np.pi + (ix.astype(float) + sx) * self.step,
                -np.pi + (iy.astype(float) + sy) * self.step,
            )
        )
        periodic_error = (values - reconstructed + np.pi) % (2.0 * np.pi) - np.pi
        if float(np.max(np.abs(periodic_error))) > 2e-11:
            raise ValueError("points do not belong to the requested commensurate subgrid")
        return ix, iy

    def endpoint_flat_indices(
        self,
        ix: np.ndarray,
        iy: np.ndarray,
        origin: tuple[float, float],
        sign: int,
    ) -> np.ndarray:
        if sign not in {-1, 1}:
            raise ValueError("endpoint sign must be -1 or +1")
        sx, sy = _origin_key(origin)
        tx, ty = self.endpoint_origin(origin)
        offset_x = int(round(sx + 0.5 * sign * self.mx - tx))
        offset_y = int(round(sy + 0.5 * sign * self.my - ty))
        target_x = (np.asarray(ix, dtype=np.int64) + offset_x) % self.nk
        target_y = (np.asarray(iy, dtype=np.int64) + offset_y) % self.nk
        return target_x * self.nk + target_y


def build_commensurate_dwave_bdg_cache(
    context: DWaveStaticIntegrandContext,
    *,
    nk: int,
    mx: int,
    my: int,
    origins: tuple[tuple[float, float], ...],
    chunk_size: int,
    max_points: int,
) -> CommensurateDWaveBdGCache:
    """Diagonalize each required shifted tensor subgrid exactly once."""

    started = time.perf_counter()
    cached: dict[tuple[float, float], CachedSubgridBands] = {}
    for origin in origins:
        key = _origin_key(origin)
        if key in cached:
            continue
        grid = CommensuratePeriodicGrid(
            nk=nk,
            mx=mx,
            my=my,
            shift_x=key[0],
            shift_y=key[1],
            max_points=max_points,
        )
        subgrid_started = time.perf_counter()
        energy_chunks: list[np.ndarray] = []
        state_chunks: list[np.ndarray] = []
        occupation_chunks: list[np.ndarray] = []
        for points in grid.iter_point_chunks(chunk_size):
            energies, states = _batched_eigensystems(
                context.spec,
                context.ansatz,
                context.pairing_params,
                points,
            )
            occupations = fermi_function(
                energies,
                context.config.fermi_level_eV,
                context.config.temperature_eV,
            )
            energy_chunks.append(energies)
            state_chunks.append(states)
            occupation_chunks.append(np.asarray(occupations, dtype=float))
        cached[key] = CachedSubgridBands(
            origin=key,
            energies=np.concatenate(energy_chunks, axis=0),
            states=np.concatenate(state_chunks, axis=0),
            occupations=np.concatenate(occupation_chunks, axis=0),
            build_wall_seconds=time.perf_counter() - subgrid_started,
        )

    cache = CommensurateDWaveBdGCache(
        context=context,
        nk=int(nk),
        mx=int(mx),
        my=int(my),
        subgrids=cached,
        chunk_size=int(chunk_size),
        build_wall_seconds=time.perf_counter() - started,
    )
    for origin in origins:
        cache.subgrid(cache.endpoint_origin(origin))
    return cache


class CachedCommensurateDWaveContext:
    """Primitive evaluator backed by exact commensurate index maps."""

    def __init__(
        self,
        cache: CommensurateDWaveBdGCache,
        origin: tuple[float, float],
    ) -> None:
        self.cache = cache
        self.origin = _origin_key(origin)
        self.spec = cache.context.spec
        self.ansatz = cache.context.ansatz
        self.q_model = cache.context.q_model
        self.config = cache.context.config
        self.pairing_params = cache.context.pairing_params
        self.options = cache.context.options
        self.density = cache.context.density
        self.delta0_eV = cache.context.delta0_eV

    def evaluate_complex(self, k_points: np.ndarray) -> np.ndarray:
        points = np.asarray(k_points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("k_points must have shape (n, 2) with finite values")

        ix, iy = self.cache.source_indices(points, self.origin)
        mid_flat = ix * self.cache.nk + iy
        minus_flat = self.cache.endpoint_flat_indices(
            ix, iy, self.origin, -1
        )
        plus_flat = self.cache.endpoint_flat_indices(ix, iy, self.origin, 1)
        midpoint = self.cache.subgrid(self.origin)
        endpoint = self.cache.subgrid(self.cache.endpoint_origin(self.origin))

        return self._evaluate_precomputed(
            points,
            midpoint.energies[mid_flat],
            midpoint.states[mid_flat],
            midpoint.occupations[mid_flat],
            endpoint.energies[minus_flat],
            endpoint.states[minus_flat],
            endpoint.occupations[minus_flat],
            endpoint.energies[plus_flat],
            endpoint.states[plus_flat],
            endpoint.occupations[plus_flat],
        )

    def _evaluate_precomputed(
        self,
        points: np.ndarray,
        energies_mid: np.ndarray,
        states_mid: np.ndarray,
        occupations_mid: np.ndarray,
        energies_minus: np.ndarray,
        states_minus: np.ndarray,
        occupations_minus: np.ndarray,
        energies_plus: np.ndarray,
        states_plus: np.ndarray,
        occupations_plus: np.ndarray,
    ) -> np.ndarray:
        qx, qy = float(self.q_model[0]), float(self.q_model[1])
        spec, ansatz = self.spec, self.ansatz
        amp, opts = self.pairing_params, self.options
        result = np.zeros((points.shape[0], _COMPLEX_WIDTH), dtype=complex)

        for index, (kx_value, ky_value) in enumerate(points):
            kx, ky = float(kx_value), float(ky_value)
            mid_states = states_mid[index]
            minus_states = states_minus[index]
            plus_states = states_plus[index]
            mid_occupations = occupations_mid[index]
            minus_occupations = occupations_minus[index]
            plus_occupations = occupations_plus[index]
            thermal_density = _thermal_density(mid_states, mid_occupations)

            vx = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "x", opts.current_vertex
            )
            vy = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "y", opts.current_vertex
            )
            collective_vertices = tuple(
                ansatz.collective_vertices(kx, ky, qx, qy, amp)
            )
            if len(collective_vertices) != _COLLECTIVE_CHANNELS:
                raise ValueError("cached integrand requires two collective channels")
            all_vertices = np.stack(
                (self.density, vx, vy, *collective_vertices), axis=0
            )
            all_band = _band_transform_stack(
                minus_states, all_vertices, plus_states
            )
            source_band = all_band[:_EM_CHANNELS]
            collective_band = all_band[_EM_CHANNELS:]
            observable_band = _EM_OBSERVABLE_SIGNS[:, None, None] * source_band
            left_band = np.concatenate((observable_band, collective_band), axis=0)
            right_band = np.concatenate((source_band, collective_band), axis=0)
            factor = _static_factor_matrix(
                energies_minus[index],
                minus_occupations,
                energies_plus[index],
                plus_occupations,
                self.config,
            )
            unified = _unified_contraction(factor, left_band, right_band)

            direct = np.zeros((_EM_CHANNELS, _EM_CHANNELS), dtype=complex)
            for i, direction_i in enumerate(("x", "y")):
                for j, direction_j in enumerate(("x", "y")):
                    contact = bdg_contact_vertex_from_spec(
                        spec,
                        kx,
                        ky,
                        qx,
                        qy,
                        direction_i,
                        direction_j,
                        opts.current_vertex,
                    )
                    direct[1 + i, 1 + j] = -_thermal_expectation_from_density(
                        thermal_density, contact
                    )

            delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
            phase_direct_plus = _thermal_expectation_from_density(
                thermal_density, phase_phase_direct_vertex(delta_theta)
            )
            collective_zero = tuple(
                ansatz.collective_vertices(kx, ky, 0.0, 0.0, amp)
            )
            if len(collective_zero) != _COLLECTIVE_CHANNELS:
                raise ValueError("cached counterterm requires two collective channels")
            eta2_band = (
                mid_states.conjugate().T
                @ np.asarray(collective_zero[1], dtype=complex)
                @ mid_states
            )
            midpoint_factor = _static_factor_matrix(
                energies_mid[index],
                mid_occupations,
                energies_mid[index],
                mid_occupations,
                self.config,
            )
            eta2_bubble = 0.5 * np.sum(
                midpoint_factor * eta2_band * np.conjugate(eta2_band)
            )
            counterterm = -complex(eta2_bubble) * np.eye(
                _COLLECTIVE_CHANNELS, dtype=complex
            )

            occupation_difference = (
                np.asarray(minus_occupations, dtype=float)[:, None]
                - np.asarray(plus_occupations, dtype=float)[None, :]
            )
            ward_equal = _ward_equal_contraction(
                occupation_difference, source_band
            )
            ward_delta_v = np.zeros(_EM_CHANNELS, dtype=complex)
            for j, direction in enumerate(("x", "y"), start=1):
                vertex_plus = bdg_vector_vertex_from_spec(
                    spec,
                    kx + 0.5 * qx,
                    ky + 0.5 * qy,
                    qx,
                    qy,
                    direction,
                    opts.current_vertex,
                )
                vertex_minus = bdg_vector_vertex_from_spec(
                    spec,
                    kx - 0.5 * qx,
                    ky - 0.5 * qy,
                    qx,
                    qy,
                    direction,
                    opts.current_vertex,
                )
                ward_delta_v[j] = _thermal_expectation_from_density(
                    thermal_density, vertex_plus - vertex_minus
                )

            ward_contact = np.zeros(_EM_CHANNELS, dtype=complex)
            for i, qi in enumerate((qx, qy)):
                for j in range(2):
                    ward_contact[1 + j] += qi * direct[1 + i, 1 + j]

            row = result[index]
            row[_UNIFIED_SLICE] = unified.reshape(-1)
            row[_DIRECT_SLICE] = direct.reshape(-1)
            row[_PHASE_DIRECT_SLICE] = phase_direct_plus
            row[_COUNTERTERM_SLICE] = counterterm.reshape(-1)
            row[_WARD_EQUAL_SLICE] = ward_equal
            row[_WARD_DELTA_V_SLICE] = ward_delta_v
            row[_WARD_CONTACT_SLICE] = ward_contact

        return result


__all__ = [
    "CachedCommensurateDWaveContext",
    "CachedSubgridBands",
    "CommensurateDWaveBdGCache",
    "build_commensurate_dwave_bdg_cache",
]
