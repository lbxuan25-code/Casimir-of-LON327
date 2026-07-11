"""Ward-closed pointwise primitives for global d-wave adaptive integration.

The vector integrand contains every primitive quantity needed by the exact-static
finite-q response: the unified electromagnetic/collective bubble, contact term,
Goldstone counterterm, phase direct term, and all three pieces of the analytic Ward
RHS.  A single vector-valued adaptive rule therefore supplies common nodes and
weights to every channel.  The amplitude/phase Schur complement is applied only
after the complete Brillouin-zone integral has been assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lno327.bdg.finite_q import density_vertex, phase_phase_direct_vertex
from lno327.response.config import KuboConfig
from lno327.response.finite_q import BdGFiniteQResponseComponents, vertex_band
from lno327.response.finite_q_bdg import (
    _check_options,
    _finalize_components,
    _pairing_params_from_inputs,
    bdg_contact_vertex_from_spec,
    bdg_eigensystem_from_model_pairing,
    bdg_vector_vertex_from_spec,
    require_peierls_finite_q_support,
)
from lno327.response.occupations import fermi_function
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.iterated_adaptive import (
    IntegrationOrder,
    IteratedAdaptiveOptions,
    IteratedAdaptiveResult,
    iterated_adaptive_integral,
)

_EM_CHANNELS = 3
_COLLECTIVE_CHANNELS = 2
_UNIFIED_CHANNELS = _EM_CHANNELS + _COLLECTIVE_CHANNELS
_EM_OBSERVABLE_SIGNS = np.asarray([1.0, -1.0, -1.0], dtype=float)

_UNIFIED_SLICE = slice(0, 25)
_DIRECT_SLICE = slice(25, 34)
_PHASE_DIRECT_SLICE = slice(34, 35)
_COUNTERTERM_SLICE = slice(35, 39)
_WARD_EQUAL_SLICE = slice(39, 42)
_WARD_DELTA_V_SLICE = slice(42, 45)
_WARD_CONTACT_SLICE = slice(45, 48)
_COMPLEX_WIDTH = 48
_REAL_WIDTH = 2 * _COMPLEX_WIDTH
_BZ_NORMALIZATION = 1.0 / (2.0 * np.pi) ** 2


def _static_factor_matrix(
    energies_minus: np.ndarray,
    occupations_minus: np.ndarray,
    energies_plus: np.ndarray,
    occupations_plus: np.ndarray,
    config: KuboConfig,
) -> np.ndarray:
    em = np.asarray(energies_minus, dtype=float)[:, None]
    en = np.asarray(energies_plus, dtype=float)[None, :]
    fm = np.asarray(occupations_minus, dtype=float)[:, None]
    fn = np.asarray(occupations_plus, dtype=float)[None, :]
    delta_e = em - en
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (fm - fn) / delta_e

    degenerate = np.abs(delta_e) < float(config.eta_eV)
    if np.any(degenerate):
        midpoint = 0.5 * (em + en) - float(config.fermi_level_eV)
        if float(config.temperature_eV) <= 0.0:
            width = max(float(config.eta_eV), 1e-12)
            derivative = -width / (np.pi * (midpoint * midpoint + width * width))
        else:
            x = np.clip(
                midpoint / (2.0 * float(config.temperature_eV)), -350.0, 350.0
            )
            derivative = -1.0 / (
                4.0 * float(config.temperature_eV) * np.cosh(x) ** 2
            )
        factor = np.where(degenerate, derivative, factor)
    return np.asarray(factor, dtype=complex)


def _thermal_expectation(
    states: np.ndarray, occupations: np.ndarray, vertex: np.ndarray
) -> complex:
    vertex_band_matrix = (
        np.asarray(states, dtype=complex).conjugate().T
        @ np.asarray(vertex, dtype=complex)
        @ np.asarray(states, dtype=complex)
    )
    return complex(
        0.5
        * np.sum(
            np.asarray(occupations, dtype=float) * np.diag(vertex_band_matrix)
        )
    )


def pack_complex_vector(value: np.ndarray) -> np.ndarray:
    """Pack one or many complex primitive vectors into a real vector contract."""

    array = np.asarray(value, dtype=complex)
    if array.shape[-1] != _COMPLEX_WIDTH:
        raise ValueError(
            f"complex primitive vector must end in width {_COMPLEX_WIDTH}, got {array.shape}"
        )
    return np.concatenate((array.real, array.imag), axis=-1)


def unpack_complex_vector(value: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_complex_vector`."""

    array = np.asarray(value, dtype=float)
    if array.shape[-1] != _REAL_WIDTH:
        raise ValueError(
            f"real primitive vector must end in width {_REAL_WIDTH}, got {array.shape}"
        )
    return np.asarray(
        array[..., :_COMPLEX_WIDTH] + 1j * array[..., _COMPLEX_WIDTH:],
        dtype=complex,
    )


@dataclass(frozen=True)
class DWaveStaticIntegrandContext:
    """Immutable model state for arbitrary exact-static BZ point evaluation."""

    spec: object
    ansatz: object
    q_model: np.ndarray
    config: KuboConfig
    pairing_params: object
    options: object
    density: np.ndarray
    delta0_eV: float

    def __post_init__(self) -> None:
        q = np.asarray(self.q_model, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("q_model must be a finite vector with shape (2,)")
        if float(self.config.omega_eV) != 0.0:
            raise ValueError("global static adaptive integration requires omega_eV == 0")
        if str(self.options.current_vertex) != "peierls":
            raise ValueError("adaptive production prototype requires current_vertex='peierls'")
        if str(self.options.collective_mode) != "amplitude_phase":
            raise ValueError(
                "adaptive production prototype requires collective_mode='amplitude_phase'"
            )
        if str(self.options.collective_counterterm) != "goldstone_gap_equation":
            raise ValueError(
                "adaptive production prototype requires the Goldstone gap-equation counterterm"
            )
        if float(self.delta0_eV) <= 0.0 or not np.isfinite(float(self.delta0_eV)):
            raise ValueError("delta0_eV must be finite and positive")
        object.__setattr__(self, "q_model", q)
        object.__setattr__(self, "density", np.asarray(self.density, dtype=complex))

    def evaluate_complex(self, k_points: np.ndarray) -> np.ndarray:
        """Return unweighted primitive densities at arbitrary BZ points."""

        points = np.asarray(k_points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("k_points must have shape (n, 2) with finite values")

        result = np.zeros((points.shape[0], _COMPLEX_WIDTH), dtype=complex)
        qx, qy = float(self.q_model[0]), float(self.q_model[1])
        spec, ansatz = self.spec, self.ansatz
        amp, opts = self.pairing_params, self.options

        for index, (kx_value, ky_value) in enumerate(points):
            kx, ky = float(kx_value), float(ky_value)

            pairing_mid = ansatz.mean_pairing(kx, ky, amp)
            bands_mid = bdg_eigensystem_from_model_pairing(
                spec, kx, ky, pairing_mid
            )
            occupations_mid = fermi_function(
                bands_mid.energies,
                self.config.fermi_level_eV,
                self.config.temperature_eV,
            )

            pairing_minus = ansatz.mean_pairing(
                kx - 0.5 * qx, ky - 0.5 * qy, amp
            )
            pairing_plus = ansatz.mean_pairing(
                kx + 0.5 * qx, ky + 0.5 * qy, amp
            )
            bands_minus = bdg_eigensystem_from_model_pairing(
                spec, kx - 0.5 * qx, ky - 0.5 * qy, pairing_minus
            )
            bands_plus = bdg_eigensystem_from_model_pairing(
                spec, kx + 0.5 * qx, ky + 0.5 * qy, pairing_plus
            )
            occupations_minus = fermi_function(
                bands_minus.energies,
                self.config.fermi_level_eV,
                self.config.temperature_eV,
            )
            occupations_plus = fermi_function(
                bands_plus.energies,
                self.config.fermi_level_eV,
                self.config.temperature_eV,
            )

            vx = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "x", opts.current_vertex
            )
            vy = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "y", opts.current_vertex
            )
            source_band = np.stack(
                [
                    vertex_band(bands_minus.states, vertex, bands_plus.states)
                    for vertex in (self.density, vx, vy)
                ],
                axis=0,
            )
            observable_band = _EM_OBSERVABLE_SIGNS[:, None, None] * source_band

            collective_vertices = tuple(
                ansatz.collective_vertices(kx, ky, qx, qy, amp)
            )
            if len(collective_vertices) != _COLLECTIVE_CHANNELS:
                raise ValueError("adaptive integrand requires exactly two collective channels")
            collective_band = np.stack(
                [
                    vertex_band(bands_minus.states, vertex, bands_plus.states)
                    for vertex in collective_vertices
                ],
                axis=0,
            )
            left_band = np.concatenate((observable_band, collective_band), axis=0)
            right_band = np.concatenate((source_band, collective_band), axis=0)
            factor = _static_factor_matrix(
                bands_minus.energies,
                occupations_minus,
                bands_plus.energies,
                occupations_plus,
                self.config,
            )
            unified = 0.5 * np.einsum(
                "mn,amn,bmn->ab",
                factor,
                left_band,
                np.conjugate(right_band),
                optimize=True,
            )

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
                    direct[1 + i, 1 + j] = -_thermal_expectation(
                        bands_mid.states, occupations_mid, contact
                    )

            delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
            phase_direct_plus = _thermal_expectation(
                bands_mid.states,
                occupations_mid,
                phase_phase_direct_vertex(delta_theta),
            )

            collective_zero = tuple(ansatz.collective_vertices(kx, ky, 0.0, 0.0, amp))
            if len(collective_zero) != _COLLECTIVE_CHANNELS:
                raise ValueError("adaptive counterterm requires exactly two collective channels")
            eta2_band = (
                bands_mid.states.conjugate().T
                @ np.asarray(collective_zero[1], dtype=complex)
                @ bands_mid.states
            )
            midpoint_factor = _static_factor_matrix(
                bands_mid.energies,
                occupations_mid,
                bands_mid.energies,
                occupations_mid,
                self.config,
            )
            eta2_bubble = 0.5 * np.sum(
                midpoint_factor * eta2_band * np.conjugate(eta2_band)
            )
            counterterm = -complex(eta2_bubble) * np.eye(
                _COLLECTIVE_CHANNELS, dtype=complex
            )

            occupation_difference = (
                np.asarray(occupations_minus, dtype=float)[:, None]
                - np.asarray(occupations_plus, dtype=float)[None, :]
            )
            ward_equal = 0.5 * np.einsum(
                "mn,mn,jmn->j",
                occupation_difference,
                source_band[0],
                np.conjugate(source_band),
                optimize=True,
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
                ward_delta_v[j] = _thermal_expectation(
                    bands_mid.states,
                    occupations_mid,
                    vertex_plus - vertex_minus,
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

    def evaluate_real(self, kx: float, ky: float) -> np.ndarray:
        """Scalar callback used by the nested adaptive quadrature."""

        complex_value = self.evaluate_complex(np.asarray([[kx, ky]], dtype=float))[0]
        return np.asarray(pack_complex_vector(complex_value), dtype=float)


@dataclass(frozen=True)
class DWaveAdaptiveStaticResult:
    """Integrated primitive response for one nesting order."""

    components: BdGFiniteQResponseComponents
    rhs: PrimitiveWardRHS
    quadrature: IteratedAdaptiveResult
    primitive_vector: np.ndarray
    primitive_metadata: Mapping[str, Any]


def build_dwave_static_integrand_context(
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    config: KuboConfig,
    pairing_params: object | None = None,
    options: object | None = None,
) -> DWaveStaticIntegrandContext:
    """Build a pointwise evaluator matching the optimized periodic-grid contract."""

    opts = options or FiniteQEngineOptions()
    _check_options(opts)
    require_peierls_finite_q_support(spec)
    amp = _pairing_params_from_inputs(spec, pairing_params)
    delta0 = float(getattr(amp, "delta0_eV", 0.0))
    dim = int(np.asarray(spec.normal_hamiltonian(0.0, 0.0)).shape[0])
    return DWaveStaticIntegrandContext(
        spec=spec,
        ansatz=ansatz,
        q_model=np.asarray(q_model, dtype=float),
        config=config,
        pairing_params=amp,
        options=opts,
        density=density_vertex(dim),
        delta0_eV=delta0,
    )


def assemble_dwave_static_primitives(
    context: DWaveStaticIntegrandContext,
    primitive_vector: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BdGFiniteQResponseComponents, PrimitiveWardRHS, dict[str, Any]]:
    """Assemble one exact-static response after all primitive integrals are complete."""

    vector = np.asarray(primitive_vector, dtype=complex).reshape(-1)
    if vector.shape != (_COMPLEX_WIDTH,) or not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
        raise ValueError(f"primitive_vector must be a finite complex vector of width {_COMPLEX_WIDTH}")

    unified = vector[_UNIFIED_SLICE].reshape(_UNIFIED_CHANNELS, _UNIFIED_CHANNELS)
    direct = vector[_DIRECT_SLICE].reshape(_EM_CHANNELS, _EM_CHANNELS)
    phase_direct_plus = complex(vector[_PHASE_DIRECT_SLICE][0])
    counterterm = vector[_COUNTERTERM_SLICE].reshape(
        _COLLECTIVE_CHANNELS, _COLLECTIVE_CHANNELS
    )
    ward_equal = vector[_WARD_EQUAL_SLICE]
    ward_delta_v = vector[_WARD_DELTA_V_SLICE]
    ward_contact = vector[_WARD_CONTACT_SLICE]

    bubble = unified[:3, :3]
    em_collective_left = unified[:3, 3:5]
    collective_em_right = unified[3:5, :3]
    collective_bubble = unified[3:5, 3:5]
    delta0 = float(context.delta0_eV)
    phase_left = delta0 * em_collective_left[:, 1]
    phase_right = delta0 * collective_em_right[1, :]
    phase_bubble = np.asarray(
        [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
    )

    components = _finalize_components(
        ansatz=context.ansatz,
        opts=context.options,
        shared_eigenbasis_q0=False,
        shared_eigenbasis_q0_tolerance=1e-14,
        collective_mode="amplitude_phase",
        collective_mode_disabled_reason=None,
        bubble=bubble,
        direct=direct,
        phase_left=phase_left,
        phase_right=phase_right,
        phase_phase_bubble_matrix=phase_bubble,
        phase_phase_direct_plus=phase_direct_plus,
        phase_phase_direct_minus=-phase_direct_plus,
        collective_bubble=collective_bubble,
        collective_counterterm_matrix=counterterm,
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        config=context.config,
        q=context.q_model,
        workspace_evaluation=True,
    )
    component_metadata = dict(components.metadata)
    component_metadata.update(
        {
            "integration_strategy": "global_iterated_adaptive_quad_vec",
            "primitive_vector_integrated_before_schur": True,
            "all_primitive_channels_share_adaptive_nodes": True,
            **dict(metadata or {}),
        }
    )
    from dataclasses import replace

    components = replace(components, metadata=component_metadata)
    ward_rhs = ward_equal - ward_delta_v + ward_contact
    rhs = PrimitiveWardRHS(
        left=ward_rhs,
        right=ward_rhs.copy(),
        q_model=context.q_model,
        xi_eV=0.0,
        delta0_eV=delta0,
        metadata={
            "convention": "primitive_crystal_xy_rhs_aware",
            "basis": "crystal_A0_xy",
            "formula": "R_S = equal_forward - delta_v_mid + qM_mid",
            "source": "global iterated adaptive primitive vector",
            "equal_forward": ward_equal.copy(),
            "delta_v_mid": ward_delta_v.copy(),
            "qM_mid": ward_contact.copy(),
            "all_terms_share_adaptive_nodes": True,
            **dict(metadata or {}),
        },
    )
    primitive_metadata = {
        "ward_equal_forward": ward_equal,
        "ward_delta_v_mid": ward_delta_v,
        "ward_qM_mid": ward_contact,
        "phase_phase_direct_plus": phase_direct_plus,
        "bz_normalization": _BZ_NORMALIZATION,
        **dict(metadata or {}),
    }
    return components, rhs, primitive_metadata


def integrate_dwave_static_order(
    context: DWaveStaticIntegrandContext,
    *,
    order: IntegrationOrder,
    options: IteratedAdaptiveOptions,
) -> DWaveAdaptiveStaticResult:
    """Run one global nesting order and assemble exactly one final Schur complement."""

    quadrature = iterated_adaptive_integral(
        lambda kx, ky: _BZ_NORMALIZATION * context.evaluate_real(kx, ky),
        order=order,
        options=options,
    )
    primitive_vector = unpack_complex_vector(quadrature.value)
    metadata = {
        "integration_order": str(order),
        "adaptive_error_estimate": float(quadrature.error_estimate),
        "adaptive_point_evaluations": int(quadrature.point_evaluations),
        "adaptive_success": bool(quadrature.success),
    }
    components, rhs, primitive_metadata = assemble_dwave_static_primitives(
        context,
        primitive_vector,
        metadata=metadata,
    )
    return DWaveAdaptiveStaticResult(
        components=components,
        rhs=rhs,
        quadrature=quadrature,
        primitive_vector=np.asarray(primitive_vector, dtype=complex),
        primitive_metadata=primitive_metadata,
    )


__all__ = [
    "DWaveAdaptiveStaticResult",
    "DWaveStaticIntegrandContext",
    "assemble_dwave_static_primitives",
    "build_dwave_static_integrand_context",
    "integrate_dwave_static_order",
    "pack_complex_vector",
    "unpack_complex_vector",
]
