from __future__ import annotations

import numpy as np

from validation.lib.dwave_shift_spatial import SpatialDiagnosticConfig, evaluate_shift_spatial
from validation.lib.dwave_shift_spectrum import (
    aggregate_rule_spectrum,
    combined_spectral_indicators,
    periodic_component_stats,
    pointwise_spectral_indicators,
    spearman_correlation_rows,
    spectral_score_fields,
    top_fraction_rows,
)


def _config() -> SpatialDiagnosticConfig:
    return SpatialDiagnosticConfig(
        base_nk=2,
        qx=0.03,
        qy=0.02,
        temperature_K=10.0,
        delta0_eV=0.1,
        eta_eV=1e-8,
    )


def _key(shift) -> tuple[float, float]:
    value = np.asarray(shift, dtype=float)
    return round(float(value[0]), 15), round(float(value[1]), 15)


def test_pointwise_spectral_indicators_are_finite_and_aligned():
    result = evaluate_shift_spatial(
        _config(), np.asarray([0.5, 1.0 / 3.0]), keep_workspace=True
    )
    indicators = pointwise_spectral_indicators(result["workspace"])
    assert indicators
    for values in indicators.values():
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert np.all(values >= 0.0)
    assert np.all(
        indicators["shifted_min_abs_eV"]
        <= np.minimum(
            indicators["shifted_minus_min_abs_eV"],
            indicators["shifted_plus_min_abs_eV"],
        )
        + 1e-15
    )


def test_rule_spectrum_aggregation_preserves_extrema_and_builds_contrasts():
    shifts = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    cache = {
        _key(shifts[0]): {
            "midpoint_min_abs_eV": np.asarray([0.3, 0.4]),
            "shifted_minus_min_abs_eV": np.asarray([0.2, 0.3]),
            "shifted_plus_min_abs_eV": np.asarray([0.4, 0.2]),
            "shifted_min_abs_eV": np.asarray([0.2, 0.2]),
            "transition_min_gap_eV": np.asarray([0.1, 0.2]),
            "pair_min_energy_eV": np.asarray([0.5, 0.6]),
            "max_abs_kubo_factor_eV_inv": np.asarray([2.0, 3.0]),
            "max_abs_occupation_difference": np.asarray([0.8, 0.9]),
            "kubo_peak_transition_gap_eV": np.asarray([0.12, 0.22]),
            "kubo_peak_pair_energy_eV": np.asarray([0.52, 0.62]),
        },
        _key(shifts[1]): {
            "midpoint_min_abs_eV": np.asarray([0.1, 0.5]),
            "shifted_minus_min_abs_eV": np.asarray([0.1, 0.4]),
            "shifted_plus_min_abs_eV": np.asarray([0.2, 0.3]),
            "shifted_min_abs_eV": np.asarray([0.1, 0.3]),
            "transition_min_gap_eV": np.asarray([0.05, 0.4]),
            "pair_min_energy_eV": np.asarray([0.3, 0.8]),
            "max_abs_kubo_factor_eV_inv": np.asarray([5.0, 1.0]),
            "max_abs_occupation_difference": np.asarray([0.7, 1.0]),
            "kubo_peak_transition_gap_eV": np.asarray([0.08, 0.42]),
            "kubo_peak_pair_energy_eV": np.asarray([0.32, 0.82]),
        },
    }
    rule_a = aggregate_rule_spectrum(
        shifts, [0.25, 0.75], cache, key_function=_key
    )
    assert np.allclose(rule_a["midpoint_min_abs_eV"], [0.1, 0.4])
    assert np.allclose(rule_a["max_abs_kubo_factor_eV_inv"], [5.0, 3.0])
    assert np.allclose(
        rule_a["mean_max_abs_kubo_factor_eV_inv"], [4.25, 1.5]
    )
    rule_b = {name: np.asarray(value) * 1.1 for name, value in rule_a.items()}
    combined = combined_spectral_indicators(rule_a, rule_b)
    assert np.allclose(combined["midpoint_min_abs_eV"], rule_a["midpoint_min_abs_eV"])
    assert np.allclose(
        combined["max_abs_kubo_factor_eV_inv"],
        rule_b["max_abs_kubo_factor_eV_inv"],
    )
    assert np.all(
        combined["rule_mean_contrast_max_abs_kubo_factor_eV_inv"] > 0.0
    )


def test_periodic_component_stats_connect_across_bz_boundary():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True
    mask[3, 1] = True
    stats = periodic_component_stats(mask)
    assert stats["num_components"] == 1
    assert stats["largest_component_fraction"] == 1.0
    assert stats["largest_component_span_x"] <= 0.5


def test_correlations_and_top_fraction_classification_are_well_formed():
    n = 4
    count = n * n
    x = np.linspace(0.01, 1.0, count)
    masses = {
        "k_ss": x.copy(),
        "k_seta": x[::-1].copy(),
        "k_etas": x[::-1].copy(),
        "k_etaeta": x.copy(),
        "ward_rhs": x.copy(),
    }
    indicators = {
        "midpoint_min_abs_eV": x[::-1] * 0.01,
        "shifted_minus_min_abs_eV": x[::-1] * 0.01,
        "shifted_plus_min_abs_eV": x[::-1] * 0.01,
        "shifted_min_abs_eV": x[::-1] * 0.01,
        "transition_min_gap_eV": x[::-1] * 0.01,
        "pair_min_energy_eV": x[::-1] * 0.02,
        "max_abs_kubo_factor_eV_inv": x * 100.0,
        "max_abs_occupation_difference": x,
        "kubo_peak_transition_gap_eV": x[::-1] * 0.01,
        "kubo_peak_pair_energy_eV": x[::-1] * 0.02,
        "rule_mean_contrast_midpoint_min_abs_eV": x * 1e-3,
        "rule_mean_contrast_shifted_minus_min_abs_eV": x * 1e-3,
        "rule_mean_contrast_shifted_plus_min_abs_eV": x * 1e-3,
        "rule_mean_contrast_shifted_min_abs_eV": x * 1e-3,
        "rule_mean_contrast_transition_min_gap_eV": x * 1e-3,
        "rule_mean_contrast_pair_min_energy_eV": x * 1e-3,
        "rule_mean_contrast_kubo_peak_transition_gap_eV": x * 1e-3,
        "rule_mean_contrast_kubo_peak_pair_energy_eV": x * 1e-3,
        "rule_mean_contrast_max_abs_kubo_factor_eV_inv": x * 10.0,
        "rule_mean_contrast_max_abs_occupation_difference": x * 0.1,
    }
    node_distance = np.linspace(0.0, np.pi, count)
    score_fields = spectral_score_fields(
        indicators,
        node_distance,
        cell_step=2.0 * np.pi / n,
        energy_floor_eV=1e-8,
    )
    rows = spearman_correlation_rows(masses, score_fields)
    assert len(rows) == len(masses) * len(score_fields)
    kss_kubo = next(
        row
        for row in rows
        if row["block"] == "k_ss" and row["indicator"] == "kubo_factor"
    )
    assert kss_kubo["spearman_rho"] > 0.99

    top = top_fraction_rows(
        masses,
        indicators,
        node_distance,
        base_nk=n,
        temperature_eV=1e-3,
        fractions=(0.25,),
    )
    assert len(top) == len(masses)
    for row in top:
        assert row["num_cells"] == 4
        assert 0.0 <= row["difference_mass_captured"] <= 1.0
        assert int(row["num_components"]) >= 1
