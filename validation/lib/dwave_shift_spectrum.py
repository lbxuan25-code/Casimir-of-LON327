"""Spectral and geometric diagnostics for periodic-shift sensitivity maps.

The spatial shift diagnostic identifies base Brillouin-zone cells where two
Ward-compatible complete-lattice rules disagree.  This module classifies those
cells using exact-static BdG spectral information evaluated on the same shift
samples.  It does not alter the quadrature or construct any local correction.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from lno327.response.finite_q_optimized import FiniteQQWorkspace, _vectorized_kubo_factors


_MINIMUM_METRICS = (
    "midpoint_min_abs_eV",
    "shifted_minus_min_abs_eV",
    "shifted_plus_min_abs_eV",
    "shifted_min_abs_eV",
    "transition_min_gap_eV",
    "pair_min_energy_eV",
    "kubo_peak_transition_gap_eV",
    "kubo_peak_pair_energy_eV",
)
_MAXIMUM_METRICS = (
    "max_abs_kubo_factor_eV_inv",
    "max_abs_occupation_difference",
)


def pointwise_spectral_indicators(workspace: FiniteQQWorkspace) -> dict[str, np.ndarray]:
    """Return exact-static spectral indicators for every point of one shift grid."""

    midpoint = np.asarray(workspace.material.midpoint_energies, dtype=float)
    minus = np.asarray(workspace.energies_minus, dtype=float)
    plus = np.asarray(workspace.energies_plus, dtype=float)
    occ_minus = np.asarray(workspace.occupations_minus, dtype=float)
    occ_plus = np.asarray(workspace.occupations_plus, dtype=float)
    factors = np.asarray(
        _vectorized_kubo_factors(workspace, np.asarray([0.0], dtype=float))[0],
        dtype=complex,
    )

    transition_gap = np.abs(minus[:, :, None] - plus[:, None, :])
    pair_energy = np.abs(minus[:, :, None]) + np.abs(plus[:, None, :])
    factor_abs = np.abs(factors)
    occupation_difference = np.abs(occ_minus[:, :, None] - occ_plus[:, None, :])
    flat_peak = np.argmax(factor_abs.reshape(factor_abs.shape[0], -1), axis=1)
    peak_i = flat_peak // factor_abs.shape[2]
    peak_j = flat_peak % factor_abs.shape[2]
    rows = np.arange(factor_abs.shape[0])

    result = {
        "midpoint_min_abs_eV": np.min(np.abs(midpoint), axis=1),
        "shifted_minus_min_abs_eV": np.min(np.abs(minus), axis=1),
        "shifted_plus_min_abs_eV": np.min(np.abs(plus), axis=1),
        "shifted_min_abs_eV": np.minimum(
            np.min(np.abs(minus), axis=1), np.min(np.abs(plus), axis=1)
        ),
        "transition_min_gap_eV": np.min(transition_gap, axis=(1, 2)),
        "pair_min_energy_eV": np.min(pair_energy, axis=(1, 2)),
        "max_abs_kubo_factor_eV_inv": np.max(factor_abs, axis=(1, 2)),
        "max_abs_occupation_difference": np.max(occupation_difference, axis=(1, 2)),
        "kubo_peak_transition_gap_eV": transition_gap[rows, peak_i, peak_j],
        "kubo_peak_pair_energy_eV": pair_energy[rows, peak_i, peak_j],
    }
    nk = workspace.nk
    for name, values in result.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (nk,) or not np.isfinite(array).all():
            raise RuntimeError(f"spectral indicator {name} is not a finite length-nk array")
        result[name] = array
    return result


def aggregate_rule_spectrum(
    shifts: Sequence[np.ndarray],
    weights: Sequence[float],
    cache: Mapping[tuple[float, float], Mapping[str, np.ndarray]],
    *,
    key_function,
) -> dict[str, np.ndarray]:
    """Aggregate per-shift spectral descriptors for one complete-lattice rule."""

    shift_values = list(shifts)
    weight_array = np.asarray(weights, dtype=float)
    if not shift_values or weight_array.shape != (len(shift_values),):
        raise ValueError("shifts and weights must be nonempty and aligned")
    if np.any(weight_array < 0.0) or not np.isfinite(weight_array).all():
        raise ValueError("weights must be finite and non-negative")
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= 0.0:
        raise ValueError("weights must have positive sum")
    weight_array = weight_array / weight_sum

    names = tuple(cache[key_function(shift)].keys() for shift in shift_values)
    reference_names = set(names[0])
    if any(set(value) != reference_names for value in names[1:]):
        raise ValueError("all cached shift spectra must expose the same indicators")

    result: dict[str, np.ndarray] = {}
    for name in sorted(reference_names):
        stacked = np.stack(
            [np.asarray(cache[key_function(shift)][name], dtype=float) for shift in shift_values],
            axis=0,
        )
        result[f"mean_{name}"] = np.tensordot(weight_array, stacked, axes=(0, 0))
        if name in _MINIMUM_METRICS:
            result[name] = np.min(stacked, axis=0)
        elif name in _MAXIMUM_METRICS:
            result[name] = np.max(stacked, axis=0)
        else:
            result[name] = np.tensordot(weight_array, stacked, axes=(0, 0))
    return result


def combined_spectral_indicators(
    rule_a: Mapping[str, np.ndarray],
    rule_b: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build rule-independent extrema plus rule-contrast spectral indicators."""

    if set(rule_a) != set(rule_b):
        raise ValueError("rule spectral dictionaries must expose identical keys")
    result: dict[str, np.ndarray] = {}
    for name in _MINIMUM_METRICS:
        result[name] = np.minimum(
            np.asarray(rule_a[name], dtype=float), np.asarray(rule_b[name], dtype=float)
        )
    for name in _MAXIMUM_METRICS:
        result[name] = np.maximum(
            np.asarray(rule_a[name], dtype=float), np.asarray(rule_b[name], dtype=float)
        )
    for name in _MINIMUM_METRICS + _MAXIMUM_METRICS:
        mean_name = f"mean_{name}"
        result[f"rule_mean_contrast_{name}"] = np.abs(
            np.asarray(rule_b[mean_name], dtype=float)
            - np.asarray(rule_a[mean_name], dtype=float)
        )
    return result


def spectral_score_fields(
    indicators: Mapping[str, np.ndarray],
    node_distance: np.ndarray,
    *,
    cell_step: float,
    energy_floor_eV: float,
) -> dict[str, np.ndarray]:
    """Return monotone singularity scores suitable for rank correlation."""

    floor = max(float(energy_floor_eV), np.finfo(float).tiny)
    distance_floor = max(0.5 * float(cell_step), np.finfo(float).tiny)

    def low_energy(name: str) -> np.ndarray:
        return -np.log10(np.asarray(indicators[name], dtype=float) + floor)

    def positive(name: str) -> np.ndarray:
        return np.log10(np.asarray(indicators[name], dtype=float) + np.finfo(float).tiny)

    return {
        "node_proximity": -np.log10(np.asarray(node_distance, dtype=float) + distance_floor),
        "midpoint_low_energy": low_energy("midpoint_min_abs_eV"),
        "shifted_low_energy": low_energy("shifted_min_abs_eV"),
        "transition_degeneracy": low_energy("transition_min_gap_eV"),
        "low_energy_pair": low_energy("pair_min_energy_eV"),
        "kubo_peak_low_energy": low_energy("kubo_peak_pair_energy_eV"),
        "kubo_factor": positive("max_abs_kubo_factor_eV_inv"),
        "kubo_rule_contrast": positive(
            "rule_mean_contrast_max_abs_kubo_factor_eV_inv"
        ),
        "transition_gap_rule_contrast": positive(
            "rule_mean_contrast_transition_min_gap_eV"
        ),
        "shifted_energy_rule_contrast": positive(
            "rule_mean_contrast_shifted_min_abs_eV"
        ),
    }


def spearman_correlation_rows(
    masses: Mapping[str, np.ndarray],
    score_fields: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Correlate every primitive-block difference mass with each spectral score."""

    rows: list[dict[str, Any]] = []
    for block, mass_values in masses.items():
        mass = np.asarray(mass_values, dtype=float)
        for indicator, score_values in score_fields.items():
            score = np.asarray(score_values, dtype=float)
            if mass.shape != score.shape:
                raise ValueError("mass and score fields must have aligned shapes")
            if np.allclose(mass, mass[0]) or np.allclose(score, score[0]):
                rho, pvalue = float("nan"), float("nan")
            else:
                statistic = spearmanr(mass, score, nan_policy="omit")
                rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
            rows.append(
                {
                    "block": block,
                    "indicator": indicator,
                    "spearman_rho": rho,
                    "pvalue": pvalue,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                }
            )
    return rows


def periodic_component_stats(mask: np.ndarray) -> dict[str, float | int]:
    """Describe four-neighbor connected components on a periodic square grid."""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2 or selected.shape[0] != selected.shape[1]:
        raise ValueError("mask must be a square two-dimensional array")
    n = selected.shape[0]
    total = int(np.count_nonzero(selected))
    if total == 0:
        return {
            "num_components": 0,
            "largest_component_fraction": float("nan"),
            "largest_component_span_x": float("nan"),
            "largest_component_span_y": float("nan"),
        }

    visited = np.zeros_like(selected)
    components: list[list[tuple[int, int]]] = []
    for ix, iy in zip(*np.nonzero(selected), strict=True):
        if visited[ix, iy]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(ix), int(iy))])
        visited[ix, iy] = True
        component: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for nx, ny in (((x - 1) % n, y), ((x + 1) % n, y), (x, (y - 1) % n), (x, (y + 1) % n)):
                if selected[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
        components.append(component)
    largest = max(components, key=len)

    def circular_span(indices: Sequence[int]) -> float:
        unique = np.unique(np.asarray(indices, dtype=int))
        if len(unique) <= 1:
            return 1.0 / n
        extended = np.concatenate([unique, unique[:1] + n])
        gaps = np.diff(extended)
        covered_steps = n - int(np.max(gaps)) + 1
        return min(1.0, float(covered_steps / n))

    return {
        "num_components": len(components),
        "largest_component_fraction": float(len(largest) / total),
        "largest_component_span_x": circular_span([value[0] for value in largest]),
        "largest_component_span_y": circular_span([value[1] for value in largest]),
    }


def top_fraction_rows(
    masses: Mapping[str, np.ndarray],
    indicators: Mapping[str, np.ndarray],
    node_distance: np.ndarray,
    *,
    base_nk: int,
    temperature_eV: float,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> list[dict[str, Any]]:
    """Classify the most shift-sensitive cells for every primitive block."""

    n = int(base_nk)
    step = 2.0 * np.pi / float(n)
    temperature = max(float(temperature_eV), np.finfo(float).tiny)
    kubo = np.asarray(indicators["max_abs_kubo_factor_eV_inv"], dtype=float)
    kubo_p90 = float(np.quantile(kubo, 0.90))
    kubo_p99 = float(np.quantile(kubo, 0.99))
    rows: list[dict[str, Any]] = []

    for block, mass_values in masses.items():
        mass = np.asarray(mass_values, dtype=float)
        if mass.shape != (n * n,):
            raise ValueError("mass arrays must match base_nk**2")
        order = np.argsort(mass)[::-1]
        mass_total = float(np.sum(mass))
        for fraction in fractions:
            count = max(1, min(len(order), int(np.ceil(float(fraction) * len(order)))))
            selected = order[:count]
            mask = np.zeros(len(order), dtype=bool)
            mask[selected] = True
            topology = periodic_component_stats(mask.reshape(n, n))

            def selected_fraction(condition: np.ndarray) -> float:
                return float(np.mean(np.asarray(condition, dtype=bool)[selected]))

            row: dict[str, Any] = {
                "block": block,
                "top_area_fraction": float(count / len(order)),
                "num_cells": count,
                "difference_mass_captured": (
                    float(np.sum(mass[selected]) / mass_total) if mass_total > 0.0 else float("nan")
                ),
                "node_within_2_cells_fraction": selected_fraction(node_distance <= 2.0 * step),
                "node_within_4_cells_fraction": selected_fraction(node_distance <= 4.0 * step),
                "midpoint_energy_le_2T_fraction": selected_fraction(
                    indicators["midpoint_min_abs_eV"] <= 2.0 * temperature
                ),
                "shifted_energy_le_2T_fraction": selected_fraction(
                    indicators["shifted_min_abs_eV"] <= 2.0 * temperature
                ),
                "shifted_energy_le_5T_fraction": selected_fraction(
                    indicators["shifted_min_abs_eV"] <= 5.0 * temperature
                ),
                "transition_gap_le_2T_fraction": selected_fraction(
                    indicators["transition_min_gap_eV"] <= 2.0 * temperature
                ),
                "transition_gap_le_5T_fraction": selected_fraction(
                    indicators["transition_min_gap_eV"] <= 5.0 * temperature
                ),
                "pair_energy_le_4T_fraction": selected_fraction(
                    indicators["pair_min_energy_eV"] <= 4.0 * temperature
                ),
                "kubo_above_global_p90_fraction": selected_fraction(kubo >= kubo_p90),
                "kubo_above_global_p99_fraction": selected_fraction(kubo >= kubo_p99),
                "median_node_distance": float(np.median(node_distance[selected])),
                "median_shifted_min_abs_eV": float(
                    np.median(np.asarray(indicators["shifted_min_abs_eV"])[selected])
                ),
                "median_transition_min_gap_eV": float(
                    np.median(np.asarray(indicators["transition_min_gap_eV"])[selected])
                ),
                "median_pair_min_energy_eV": float(
                    np.median(np.asarray(indicators["pair_min_energy_eV"])[selected])
                ),
                "median_max_abs_kubo_factor_eV_inv": float(np.median(kubo[selected])),
                **topology,
            }
            rows.append(row)
    return rows


__all__ = [
    "aggregate_rule_spectrum",
    "combined_spectral_indicators",
    "periodic_component_stats",
    "pointwise_spectral_indicators",
    "spearman_correlation_rows",
    "spectral_score_fields",
    "top_fraction_rows",
]
