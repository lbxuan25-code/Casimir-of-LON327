"""Reusable exact-static point diagnostics for quadrature-method comparisons.

These helpers expose decompositions only. They are not a runnable point-convergence
surface; point-specific working/audit grid selection belongs exclusively to the
unified transverse-point sweet-spot command.
"""
from __future__ import annotations

import numpy as np

_LONGITUDINAL_COMPONENTS = {
    "k0l": (0, 1),
    "kl0": (1, 0),
    "kll": (1, 1),
    "klt": (1, 2),
    "ktl": (2, 1),
}
_COLLECTIVE_LABELS = ("eta1_amplitude", "eta2_phase")


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _scaled_static_kernel(kernel_lt: np.ndarray, energy_scale_eV: float) -> np.ndarray:
    matrix = np.array(kernel_lt, dtype=complex, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"kernel_lt must have shape (3, 3), got {matrix.shape}")
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")
    matrix[0, 0] *= energy
    matrix[1:3, 1:3] /= energy
    return matrix


def _complex_scalar_fields(prefix: str, value: complex, scale: float) -> dict[str, float]:
    scalar = complex(value)
    denominator = max(float(scale), 1e-30)
    return {
        f"{prefix}_real": float(scalar.real),
        f"{prefix}_imag": float(scalar.imag),
        f"{prefix}_abs": float(abs(scalar)),
        f"{prefix}_relative_abs": float(abs(scalar) / denominator),
    }


def longitudinal_component_diagnostics(
    kernel_lt: np.ndarray,
    energy_scale_eV: float,
) -> dict[str, float | str]:
    scaled = _scaled_static_kernel(kernel_lt, energy_scale_eV)
    scale = max(float(np.linalg.norm(scaled.real)), 1.0)
    result: dict[str, float | str] = {"static_kernel_real_scale": scale}
    relative_values: dict[str, float] = {}
    for name, (row, col) in _LONGITUDINAL_COMPONENTS.items():
        absolute = float(abs(scaled[row, col]))
        relative = absolute / scale
        result[f"scaled_abs_{name}"] = absolute
        result[f"relative_{name}"] = relative
        relative_values[name] = relative
    combined = float(np.linalg.norm(np.fromiter(relative_values.values(), dtype=float)))
    dominant = max(relative_values, key=relative_values.__getitem__)
    result["longitudinal_components_relative_norm"] = combined
    result["dominant_longitudinal_component"] = dominant.upper()
    result["dominant_longitudinal_relative"] = relative_values[dominant]
    return result


def _collective_inverse(kernel: object) -> np.ndarray:
    k_etaeta = np.asarray(getattr(kernel, "k_etaeta"), dtype=complex)
    method = str(getattr(kernel, "schur_inverse_method"))
    if method == "inv":
        return np.linalg.inv(k_etaeta)
    if method == "pinv_diagnostic":
        return np.linalg.pinv(k_etaeta)
    raise ValueError(f"unsupported Schur inverse method {method!r}")


def _collective_schur_correction(kernel: object) -> np.ndarray:
    k_seta = np.asarray(getattr(kernel, "k_seta"), dtype=complex)
    k_etas = np.asarray(getattr(kernel, "k_etas"), dtype=complex)
    return k_seta @ _collective_inverse(kernel) @ k_etas


def kll_decomposition_diagnostics(
    components: object,
    kernel: object,
    transform: np.ndarray,
    energy_scale_eV: float,
    static_scale: float,
) -> dict[str, float]:
    rotation = np.asarray(transform, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {rotation.shape}")
    bubble_xy = np.asarray(getattr(components, "bare_bubble"), dtype=complex)
    direct_xy = np.asarray(getattr(components, "direct"), dtype=complex)
    bare_total_xy = np.asarray(getattr(components, "bare_total"), dtype=complex)
    effective_xy = np.asarray(getattr(kernel, "k_eff"), dtype=complex)
    correction_xy = _collective_schur_correction(kernel)
    if not np.allclose(bubble_xy + direct_xy, bare_total_xy, rtol=2e-12, atol=2e-13):
        raise RuntimeError("K_SS decomposition does not satisfy bubble + direct = bare_total")
    if not np.allclose(
        bare_total_xy - correction_xy,
        effective_xy,
        rtol=2e-12,
        atol=2e-13,
    ):
        raise RuntimeError("Schur decomposition does not satisfy K_eff = K_SS - correction")
    pieces_xy = {
        "scaled_kll_bubble": bubble_xy,
        "scaled_kll_direct": direct_xy,
        "scaled_kll_bare_total": bare_total_xy,
        "scaled_kll_collective_correction": correction_xy,
        "scaled_kll_effective": effective_xy,
    }
    values: dict[str, complex] = {}
    result: dict[str, float] = {}
    for prefix, matrix_xy in pieces_xy.items():
        matrix_lt = rotation @ matrix_xy @ rotation.T
        value = complex(_scaled_static_kernel(matrix_lt, energy_scale_eV)[1, 1])
        values[prefix] = value
        result.update(_complex_scalar_fields(prefix, value, static_scale))
    bubble = values["scaled_kll_bubble"]
    direct = values["scaled_kll_direct"]
    bare_total = values["scaled_kll_bare_total"]
    correction = values["scaled_kll_collective_correction"]
    effective = values["scaled_kll_effective"]
    result.update(
        {
            "kll_bubble_direct_cancellation_ratio": float(
                abs(bare_total) / max(abs(bubble), abs(direct), 1e-30)
            ),
            "kll_schur_cancellation_ratio": float(
                abs(effective) / max(abs(bare_total), abs(correction), 1e-30)
            ),
            "kll_bubble_direct_closure_abs": float(abs(bubble + direct - bare_total)),
            "kll_schur_closure_abs": float(abs(bare_total - correction - effective)),
        }
    )
    return result


def collective_channel_diagnostics(
    kernel: object,
    transform: np.ndarray,
    energy_scale_eV: float,
    static_scale: float,
) -> dict[str, float | str]:
    rotation = np.asarray(transform, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {rotation.shape}")
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")
    k_seta = np.asarray(getattr(kernel, "k_seta"), dtype=complex)
    k_etaeta = np.asarray(getattr(kernel, "k_etaeta"), dtype=complex)
    k_etas = np.asarray(getattr(kernel, "k_etas"), dtype=complex)
    if k_seta.shape != (3, 2) or k_etaeta.shape != (2, 2) or k_etas.shape != (2, 3):
        raise ValueError("collective channel diagnostics require 3x2, 2x2, and 2x3 blocks")
    inverse = _collective_inverse(kernel)
    k_seta_lt = rotation @ k_seta
    k_etas_lt = k_etas @ rotation.T
    left = np.asarray(k_seta_lt[1, :], dtype=complex)
    right = np.asarray(k_etas_lt[:, 1], dtype=complex)
    terms: dict[str, complex] = {}
    for a, left_label in enumerate(_COLLECTIVE_LABELS):
        for b, right_label in enumerate(_COLLECTIVE_LABELS):
            name = f"{left_label}_{right_label}"
            terms[name] = complex(left[a] * inverse[a, b] * right[b] / energy)
    correction_sum = sum(terms.values(), 0.0 + 0.0j)
    correction_xy = _collective_schur_correction(kernel)
    expected = complex(
        _scaled_static_kernel(rotation @ correction_xy @ rotation.T, energy)[1, 1]
    )
    if not np.isclose(correction_sum, expected, rtol=2e-12, atol=2e-13):
        raise RuntimeError("collective eta-channel terms do not reconstruct K_LL correction")
    singular_values = np.sort(np.asarray(np.linalg.svd(k_etaeta, compute_uv=False), dtype=float))
    eigenvalues = sorted(np.linalg.eigvals(k_etaeta), key=lambda value: abs(value))
    sv_min = float(singular_values[0])
    sv_max = float(singular_values[-1])
    result: dict[str, float | str] = {
        "collective_singular_value_min": sv_min,
        "collective_singular_value_max": sv_max,
        "collective_condition_from_svd": float(sv_max / max(sv_min, 1e-30)),
        "collective_inverse_frobenius_norm": float(np.linalg.norm(inverse)),
        "collective_channel_sum_closure_abs": float(abs(correction_sum - expected)),
    }
    result.update(_complex_scalar_fields("collective_eigenvalue_small", eigenvalues[0], sv_max))
    result.update(_complex_scalar_fields("collective_eigenvalue_large", eigenvalues[1], sv_max))
    left_scale = max(float(np.linalg.norm(left)), 1e-30)
    right_scale = max(float(np.linalg.norm(right)), 1e-30)
    for index, label in enumerate(_COLLECTIVE_LABELS):
        result.update(_complex_scalar_fields(f"raw_k_l_{label}", left[index], left_scale))
        result.update(_complex_scalar_fields(f"raw_k_{label}_l", right[index], right_scale))
    for name, value in terms.items():
        result.update(_complex_scalar_fields(f"scaled_kll_channel_{name}", value, static_scale))
    result.update(_complex_scalar_fields("scaled_kll_channel_sum", correction_sum, static_scale))
    dominant = max(terms, key=lambda name: abs(terms[name]))
    result["dominant_collective_channel"] = dominant
    result["dominant_collective_channel_relative_abs"] = float(
        abs(terms[dominant]) / max(float(static_scale), 1e-30)
    )
    result["phase_phase_fraction_of_term_norm"] = float(
        abs(terms["eta2_phase_eta2_phase"])
        / max(sum(abs(value) for value in terms.values()), 1e-30)
    )
    return result


def phase_channel_factor_diagnostics(
    components: object,
    kernel: object,
    transform: np.ndarray,
    energy_scale_eV: float,
    static_scale: float,
) -> dict[str, float]:
    rotation = np.asarray(transform, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {rotation.shape}")
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")
    collective_bubble = np.asarray(getattr(components, "collective_bubble"), dtype=complex)
    collective_counterterm = np.asarray(
        getattr(components, "collective_counterterm"), dtype=complex
    )
    collective_total = np.asarray(getattr(components, "collective_total"), dtype=complex)
    k_etaeta = np.asarray(getattr(kernel, "k_etaeta"), dtype=complex)
    if any(
        matrix.shape != (2, 2)
        for matrix in (
            collective_bubble,
            collective_counterterm,
            collective_total,
            k_etaeta,
        )
    ):
        raise ValueError("phase factor diagnostics require 2x2 collective matrices")
    if not np.allclose(
        collective_bubble + collective_counterterm,
        collective_total,
        rtol=2e-12,
        atol=2e-13,
    ):
        raise RuntimeError("collective bubble + counterterm does not reconstruct total")
    if not np.allclose(collective_total, k_etaeta, rtol=2e-12, atol=2e-13):
        raise RuntimeError("component collective_total disagrees with effective kernel k_etaeta")
    k_seta_lt = rotation @ np.asarray(getattr(kernel, "k_seta"), dtype=complex)
    k_etas_lt = np.asarray(getattr(kernel, "k_etas"), dtype=complex) @ rotation.T
    inverse = _collective_inverse(kernel)
    left_phase = complex(k_seta_lt[1, 1])
    right_phase = complex(k_etas_lt[1, 1])
    phase_bubble = complex(collective_bubble[1, 1])
    phase_counterterm = complex(collective_counterterm[1, 1])
    phase_total = complex(collective_total[1, 1])
    inverse_22 = complex(inverse[1, 1])
    scalar_inverse = complex(1.0 / phase_total) if abs(phase_total) > 0.0 else complex(np.inf)
    correction = complex(left_phase * inverse_22 * right_phase / energy)
    result: dict[str, float] = {
        "phase_collective_total_closure_abs": float(
            abs(phase_bubble + phase_counterterm - phase_total)
        ),
        "phase_inverse_scalar_mismatch_abs": float(abs(inverse_22 - scalar_inverse)),
        "phase_inverse_scalar_mismatch_relative": float(
            abs(inverse_22 - scalar_inverse)
            / max(abs(inverse_22), abs(scalar_inverse), 1e-30)
        ),
    }
    result.update(_complex_scalar_fields("raw_phase_left_coupling", left_phase, max(abs(left_phase), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_right_coupling", right_phase, max(abs(right_phase), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_collective_bubble", phase_bubble, max(abs(phase_total), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_collective_counterterm", phase_counterterm, max(abs(phase_total), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_collective_total", phase_total, max(abs(phase_total), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_inverse_22", inverse_22, max(abs(inverse_22), 1e-30)))
    result.update(_complex_scalar_fields("raw_phase_scalar_inverse", scalar_inverse, max(abs(inverse_22), abs(scalar_inverse), 1e-30)))
    result.update(_complex_scalar_fields("scaled_phase_factorized_correction", correction, static_scale))
    result["phase_factorized_correction_closure_abs"] = 0.0
    result["phase_bubble_counterterm_cancellation_ratio"] = float(
        abs(phase_total) / max(abs(phase_bubble), abs(phase_counterterm), 1e-30)
    )
    return result


def ward_side_diagnostics(side: object, prefix: str) -> dict[str, float]:
    rhs_norm = _norm(getattr(side, "primitive_rhs"))
    projection_norm = _norm(getattr(side, "collective_projection"))
    direct_norm = _norm(getattr(side, "effective_direct"))
    predicted_norm = _norm(getattr(side, "effective_predicted"))
    residual_norm = _norm(getattr(side, "effective_residual"))
    return {
        f"{prefix}_rhs_norm": rhs_norm,
        f"{prefix}_collective_projection_norm": projection_norm,
        f"{prefix}_effective_direct_norm": direct_norm,
        f"{prefix}_effective_predicted_norm": predicted_norm,
        f"{prefix}_effective_residual_norm": residual_norm,
        f"{prefix}_rhs_projection_cancellation_ratio": predicted_norm
        / max(rhs_norm, projection_norm, 1e-30),
        f"{prefix}_direct_prediction_relative_residual": residual_norm
        / max(direct_norm, predicted_norm, 1e-30),
    }


__all__ = [
    "collective_channel_diagnostics",
    "kll_decomposition_diagnostics",
    "longitudinal_component_diagnostics",
    "phase_channel_factor_diagnostics",
    "ward_side_diagnostics",
]
