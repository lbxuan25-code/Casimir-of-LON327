"""Experimental Ward-aware adapter for the canonical static d-wave quadrature.

This module leaves the 48-complex primitive response contract unchanged.  The
Gauss/adaptive command receives a translation-paired physical integrand plus
scaled diagnostic channels that participate only in adaptive error estimation.
After integration, the diagnostic tail is discarded before primitive assembly.

The adapter is installed only by :mod:`validation.commands.static`; commensurate
and other library-level integrations keep using the unmodified pointwise
primitive evaluator.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lno327.response.phase_hessian import nearest_neighbor_dwave_bond_metric
from validation.lib import dwave_static_primitives as _primitives

# The diagnostic channels are normalized by |q| and this dimensionless scale.
# With the canonical epsabs=2e-4, 5e-3 targets an absolute residual/|q| error
# near 1e-6 without changing the physical primitive vector.
WARD_DIAGNOSTIC_SCALE = 5e-3
WARD_DIAGNOSTIC_COMPLEX_WIDTH = 10
WARD_DIAGNOSTIC_REAL_WIDTH = 2 * WARD_DIAGNOSTIC_COMPLEX_WIDTH
AUGMENTED_REAL_WIDTH = _primitives._REAL_WIDTH + WARD_DIAGNOSTIC_REAL_WIDTH

_ORIGINAL_EVALUATE_REAL = _primitives.DWaveStaticIntegrandContext.evaluate_real
_ORIGINAL_UNPACK_COMPLEX_VECTOR = _primitives.unpack_complex_vector
_ORIGINAL_ASSEMBLE = _primitives.assemble_dwave_static_primitives
_INSTALLED = False


def _wrap_periodic_bz(points: np.ndarray) -> np.ndarray:
    """Map arbitrary points to the canonical ``[-pi, pi)`` Brillouin zone."""

    array = np.asarray(points, dtype=float)
    return np.asarray((array + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)


def paired_physical_primitive(
    context: _primitives.DWaveStaticIntegrandContext,
    kx: float,
    ky: float,
) -> np.ndarray:
    """Return the ``+/- q/2`` translation-paired 48-component primitive density."""

    q = np.asarray(context.q_model, dtype=float)
    center = np.asarray([float(kx), float(ky)], dtype=float)
    points = _wrap_periodic_bz(
        np.stack((center - 0.5 * q, center + 0.5 * q), axis=0)
    )
    values = np.asarray(context.evaluate_complex(points), dtype=complex)
    if values.shape != (2, _primitives._COMPLEX_WIDTH):
        raise ValueError(
            "paired primitive evaluation must return shape "
            f"(2, {_primitives._COMPLEX_WIDTH}), got {values.shape}"
        )
    return np.asarray(np.mean(values, axis=0), dtype=complex)


def ward_diagnostic_channels(
    context: _primitives.DWaveStaticIntegrandContext,
    primitive_vector: np.ndarray,
) -> np.ndarray:
    """Build linear Ward-sensitive channels used only by the error estimator.

    The diagnostics contain left/right primitive residuals (three components
    each) and left/right collective residuals (two components each).  The
    nearest-neighbour bond metric is applied only inside the diagnostic phase
    Hessian, which is algebraically equivalent to applying the constant metric
    after the complete integral.  The physical primitive vector remains
    untouched and is still postprocessed exactly once by the canonical path.
    """

    vector = np.asarray(primitive_vector, dtype=complex).reshape(-1)
    if vector.shape != (_primitives._COMPLEX_WIDTH,):
        raise ValueError(
            "primitive_vector must have width "
            f"{_primitives._COMPLEX_WIDTH}, got {vector.shape}"
        )

    unified = vector[_primitives._UNIFIED_SLICE].reshape(
        _primitives._UNIFIED_CHANNELS,
        _primitives._UNIFIED_CHANNELS,
    )
    direct = vector[_primitives._DIRECT_SLICE].reshape(
        _primitives._EM_CHANNELS,
        _primitives._EM_CHANNELS,
    )
    counterterm = vector[_primitives._COUNTERTERM_SLICE].reshape(
        _primitives._COLLECTIVE_CHANNELS,
        _primitives._COLLECTIVE_CHANNELS,
    )
    ward_rhs = (
        vector[_primitives._WARD_EQUAL_SLICE]
        - vector[_primitives._WARD_DELTA_V_SLICE]
        + vector[_primitives._WARD_CONTACT_SLICE]
    )

    k_ss = unified[:3, :3] + direct
    k_seta = unified[:3, 3:5]
    k_etas = unified[3:5, :3]
    corrected_counterterm = np.array(counterterm, dtype=complex, copy=True)
    corrected_counterterm[1, 1] *= nearest_neighbor_dwave_bond_metric(
        context.q_model
    )
    k_etaeta = unified[3:5, 3:5] + corrected_counterterm

    q = np.asarray(context.q_model, dtype=float)
    q_norm = float(np.linalg.norm(q))
    if not np.isfinite(q_norm) or q_norm <= 0.0:
        raise ValueError("Ward-aware quadrature requires finite nonzero q")
    u = np.asarray([0.0 + 0.0j, q[0], q[1]], dtype=complex)
    w = np.asarray([0.0 + 0.0j, -2j * float(context.delta0_eV)], dtype=complex)

    left_primitive = u @ k_ss + w @ k_etas - ward_rhs
    right_primitive = k_ss @ u + k_seta @ w - ward_rhs
    left_collective = u @ k_seta + w @ k_etaeta
    right_collective = k_etas @ u + k_etaeta @ w

    diagnostics = np.concatenate(
        (
            np.asarray(left_primitive, dtype=complex),
            np.asarray(right_primitive, dtype=complex),
            np.asarray(left_collective, dtype=complex),
            np.asarray(right_collective, dtype=complex),
        )
    )
    if diagnostics.shape != (WARD_DIAGNOSTIC_COMPLEX_WIDTH,):
        raise RuntimeError(
            "internal Ward diagnostic width mismatch: "
            f"expected {WARD_DIAGNOSTIC_COMPLEX_WIDTH}, got {diagnostics.shape}"
        )
    if not np.isfinite(diagnostics.real).all() or not np.isfinite(
        diagnostics.imag
    ).all():
        raise ValueError("Ward diagnostic channels must be finite")

    return np.asarray(
        diagnostics / (q_norm * WARD_DIAGNOSTIC_SCALE),
        dtype=complex,
    )


def ward_aware_real_vector(
    context: _primitives.DWaveStaticIntegrandContext,
    kx: float,
    ky: float,
) -> np.ndarray:
    """Return physical packed channels followed by scaled Ward diagnostics."""

    physical = paired_physical_primitive(context, kx, ky)
    packed_physical = np.asarray(_primitives.pack_complex_vector(physical), dtype=float)
    diagnostics = ward_diagnostic_channels(context, physical)
    packed_diagnostics = np.concatenate((diagnostics.real, diagnostics.imag))
    result = np.concatenate((packed_physical, packed_diagnostics))
    if result.shape != (AUGMENTED_REAL_WIDTH,):
        raise RuntimeError(
            "internal augmented real width mismatch: "
            f"expected {AUGMENTED_REAL_WIDTH}, got {result.shape}"
        )
    return np.asarray(result, dtype=float)


def unpack_augmented_complex_vector(value: np.ndarray) -> np.ndarray:
    """Discard Ward diagnostics and unpack only the physical primitive prefix."""

    array = np.asarray(value, dtype=float)
    width = int(array.shape[-1])
    if width == _primitives._REAL_WIDTH:
        return _ORIGINAL_UNPACK_COMPLEX_VECTOR(array)
    if width != AUGMENTED_REAL_WIDTH:
        raise ValueError(
            "real primitive vector must end in either physical width "
            f"{_primitives._REAL_WIDTH} or Ward-aware width "
            f"{AUGMENTED_REAL_WIDTH}, got {array.shape}"
        )
    return _ORIGINAL_UNPACK_COMPLEX_VECTOR(
        np.asarray(array[..., : _primitives._REAL_WIDTH], dtype=float)
    )


def assemble_with_ward_aware_metadata(
    context: _primitives.DWaveStaticIntegrandContext,
    primitive_vector: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
):
    """Inject an explicit audit trail for the experimental Gauss integrand."""

    merged = dict(metadata or {})
    if merged.get("integration_strategy") == "fixed_gauss_outer_adaptive_inner":
        merged.update(
            {
                "quadrature_physical_integrand": "paired_translation_pm_half_q",
                "quadrature_error_estimator": "physical_plus_scaled_ward_channels",
                "ward_diagnostic_complex_width": WARD_DIAGNOSTIC_COMPLEX_WIDTH,
                "ward_diagnostic_scale_after_q_normalization": WARD_DIAGNOSTIC_SCALE,
                "microscopic_primitive_evaluations_per_quadrature_point": 2,
                "ward_diagnostics_discarded_before_primitive_assembly": True,
                "physical_primitive_contract_complex_width": _primitives._COMPLEX_WIDTH,
            }
        )
    return _ORIGINAL_ASSEMBLE(
        context,
        primitive_vector,
        metadata=merged,
    )


def install_dwave_ward_aware_quadrature() -> None:
    """Install the experimental adapter for static command imports, idempotently."""

    global _INSTALLED
    if _INSTALLED:
        return
    _primitives.DWaveStaticIntegrandContext.evaluate_real = ward_aware_real_vector
    _primitives.unpack_complex_vector = unpack_augmented_complex_vector
    _primitives.assemble_dwave_static_primitives = assemble_with_ward_aware_metadata
    _INSTALLED = True


__all__ = [
    "AUGMENTED_REAL_WIDTH",
    "WARD_DIAGNOSTIC_COMPLEX_WIDTH",
    "WARD_DIAGNOSTIC_SCALE",
    "assemble_with_ward_aware_metadata",
    "install_dwave_ward_aware_quadrature",
    "paired_physical_primitive",
    "unpack_augmented_complex_vector",
    "ward_aware_real_vector",
    "ward_diagnostic_channels",
]
