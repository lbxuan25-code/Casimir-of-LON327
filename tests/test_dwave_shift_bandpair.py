from __future__ import annotations

import numpy as np

from validation.lib.dwave_shift_bandpair import (
    PAIR_BLOCKS,
    aggregate_pair_classification,
    aggregate_rule_pair_strengths,
    bandpair_mass_summary,
    dominant_pair_fields,
    normal_shifted_fs_fields,
    pair_strength_contrast,
    pointwise_bandpair_data,
)
from validation.lib.dwave_shift_spatial import SpatialDiagnosticConfig, evaluate_shift_spatial
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _config() -> SpatialDiagnosticConfig:
    return SpatialDiagnosticConfig(
        base_nk=2,
        qx=0.03,
        qy=0.02,
        temperature_K=10.0,
        delta0_eV=0.1,
        eta_eV=1e-8,
    )


def _key(shift):
    value = np.asarray(shift, dtype=float)
    return round(float(value[0]), 15), round(float(value[1]), 15)


def test_pointwise_bandpair_data_are_finite_and_normalized():
    result = evaluate_shift_spatial(_config(), np.asarray([0.5, 1.0 / 3.0]), keep_workspace=True)
    data = pointwise_bandpair_data(result["workspace"])
    nk, nb = result["workspace"].nk, result["workspace"].nb
    assert set(data["strengths"]) == set(PAIR_BLOCKS)
    for values in data["strengths"].values():
        assert values.shape == (nk, nb, nb)
        assert np.isfinite(values).all()
        assert np.all(values >= 0.0)
    for name in ("normal_band_weights_minus", "normal_band_weights_plus"):
        weights = np.asarray(data[name], dtype=float)
        assert weights.shape == (nk, nb, 2)
        assert np.allclose(np.sum(weights, axis=2), 1.0, rtol=1e-10, atol=1e-11)
    for name in ("particle_weight_minus", "particle_weight_plus"):
        values = np.asarray(data[name], dtype=float)
        assert np.all(values >= -1e-12)
        assert np.all(values <= 1.0 + 1e-12)


def test_rule_pair_aggregation_contrast_and_classification_shapes():
    shifts_a = np.asarray([[0.25, 0.25], [0.75, 0.75]])
    shifts_b = np.asarray([[0.25, 0.75], [0.75, 0.25]])
    all_shifts = np.concatenate([shifts_a, shifts_b], axis=0)
    cache = {}
    for shift in all_shifts:
        result = evaluate_shift_spatial(_config(), shift, keep_workspace=True)
        cache[_key(shift)] = pointwise_bandpair_data(result["workspace"])
    weights = np.asarray([0.5, 0.5])
    rule_a = aggregate_rule_pair_strengths(shifts_a, weights, cache, key_function=_key)
    rule_b = aggregate_rule_pair_strengths(shifts_b, weights, cache, key_function=_key)
    contrast = pair_strength_contrast(rule_a, rule_b)
    classification = aggregate_pair_classification(
        shifts_a, weights, shifts_b, weights, cache, key_function=_key
    )
    dominant = dominant_pair_fields(contrast, classification)
    nk = _config().base_nk**2
    for block in PAIR_BLOCKS:
        assert contrast[block].shape[0] == nk
        assert dominant[block]["m"].shape == (nk,)
        assert dominant[block]["n"].shape == (nk,)
        assert np.all(dominant[block]["sign_crossing_fraction"] >= 0.0)
        assert np.all(dominant[block]["sign_crossing_fraction"] <= 1.0)
    ph_sum = sum(
        classification[name]
        for name in ("ph_pp_fraction", "ph_ph_fraction", "ph_hp_fraction", "ph_hh_fraction")
    )
    assert np.allclose(ph_sum, 1.0, rtol=0.0, atol=1e-12)


def test_normal_shifted_fs_fields_and_mass_summary_are_bounded():
    n = 4
    coordinates = -np.pi + (np.arange(n, dtype=float) + 0.5) * (2.0 * np.pi / n)
    gx, gy = np.meshgrid(coordinates, coordinates, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    normal_fs = normal_shifted_fs_fields(model.spec, centers, np.asarray([0.03, 0.02]))
    assert normal_fs["normal_minus_eV"].shape == (n * n, 2)
    assert normal_fs["normal_plus_eV"].shape == (n * n, 2)
    assert normal_fs["same_normal_band_sign_crossing"].shape == (n * n,)

    masses = {name: np.linspace(1.0, 2.0, n * n) for name in ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs")}
    pair_shape = (n * n, 2, 2)
    contrast = {name: np.ones(pair_shape) for name in PAIR_BLOCKS}
    contrast["k_ss"][:, 0, 0] = 3.0
    classification = {
        "sign_crossing_fraction": np.ones(pair_shape),
        "same_normal_band_fraction": np.ones(pair_shape),
        "same_bdg_index_fraction": np.broadcast_to(np.eye(2)[None], pair_shape),
        "particle_weight_minus": np.full(pair_shape, 0.75),
        "particle_weight_plus": np.full(pair_shape, 0.75),
        "normal_00_fraction": np.ones(pair_shape),
        "normal_11_fraction": np.zeros(pair_shape),
        "normal_interband_fraction": np.zeros(pair_shape),
        "ph_pp_fraction": np.ones(pair_shape),
        "ph_ph_fraction": np.zeros(pair_shape),
        "ph_hp_fraction": np.zeros(pair_shape),
        "ph_hh_fraction": np.zeros(pair_shape),
    }
    dominant = dominant_pair_fields(contrast, classification)
    rows = bandpair_mass_summary(masses, dominant, normal_fs, top_area_fraction=0.25)
    assert len(rows) == 5
    for row in rows:
        for key, value in row.items():
            if key.endswith("fraction") or key.endswith("captured"):
                assert 0.0 <= float(value) <= 1.0
