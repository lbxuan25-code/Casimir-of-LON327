from __future__ import annotations

import numpy as np

from validation.lib.dwave_shift_signed_reconstruction import (
    actual_shift_normal_fs_data,
    aggregate_rule_signed_summaries,
    signed_pair_primitive_vectors,
    signed_pair_reconstruction_residuals,
    signed_reconstruction_residuals,
    summarize_shift_signed_reconstruction,
)
from validation.lib.dwave_shift_spatial import (
    PRIMITIVE_SLICES,
    SpatialDiagnosticConfig,
    evaluate_shift_spatial,
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


def _result():
    return evaluate_shift_spatial(
        _config(), np.asarray([0.37, 0.21], dtype=float), keep_workspace=True
    )


def test_actual_shift_normal_fs_data_are_aligned_and_finite():
    result = _result()
    workspace = result["workspace"]
    data = actual_shift_normal_fs_data(workspace)
    assert data["normal_minus_eV"].shape == (workspace.nk, 2)
    assert data["normal_plus_eV"].shape == (workspace.nk, 2)
    assert data["crossing_by_band"].shape == (workspace.nk, 2)
    assert data["any_crossing"].shape == (workspace.nk,)
    assert np.isfinite(data["normal_minus_eV"]).all()
    assert np.isfinite(data["normal_plus_eV"]).all()
    assert np.isfinite(data["minimum_shifted_abs_eV_by_band"]).all()


def test_signed_pair_sum_reproduces_all_bubble_slices():
    result = _result()
    pair_vectors = signed_pair_primitive_vectors(result["workspace"])
    pair_total = np.sum(pair_vectors, axis=(0, 1, 2))
    primitive_total = np.sum(np.asarray(result["vectors"], dtype=complex), axis=0)
    for name in (
        "bare_bubble",
        "em_collective_left",
        "collective_em_right",
        "collective_bubble",
    ):
        slc = PRIMITIVE_SLICES[name]
        assert np.allclose(pair_total[slc], primitive_total[slc], rtol=1e-11, atol=1e-12)
    assert np.allclose(pair_total[PRIMITIVE_SLICES["direct"]], 0.0, atol=0.0)
    assert np.allclose(pair_total[PRIMITIVE_SLICES["rhs_left"]], 0.0, atol=0.0)


def test_shift_summary_contains_exact_all_point_and_all_pair_audits():
    result = _result()
    summary = summarize_shift_signed_reconstruction(
        result["workspace"], result["vectors"], shell_multiples_T=(2.0, 5.0)
    )
    primitive_total = np.sum(np.asarray(result["vectors"], dtype=complex), axis=0)
    assert np.allclose(summary["spatial_sums"]["all_points"], primitive_total)
    pair_total = np.sum(signed_pair_primitive_vectors(result["workspace"]), axis=(0, 1, 2))
    assert np.allclose(summary["pair_sums"]["all_pairs"], pair_total)
    assert set(summary["spatial_sums"]) >= {
        "all_points",
        "any_normal_fs_crossing",
        "normal_band_0_fs_crossing",
        "normal_band_1_fs_crossing",
        "any_normal_fs_shell_2T",
        "any_normal_fs_shell_5T",
    }
    for value in summary["spatial_point_fractions"].values():
        assert 0.0 <= value <= 1.0
    for value in summary["pair_event_fractions"].values():
        assert 0.0 <= value <= 1.0


def test_rule_aggregation_and_zero_residual_audits():
    result = _result()
    summary = summarize_shift_signed_reconstruction(
        result["workspace"], result["vectors"], shell_multiples_T=(5.0,)
    )
    shift = np.asarray(result["shift"], dtype=float)
    key = (round(float(shift[0]), 15), round(float(shift[1]), 15))
    cache = {key: summary}
    aggregate = aggregate_rule_signed_summaries(
        [shift], [1.0], cache, key_function=lambda value: (
            round(float(value[0]), 15),
            round(float(value[1]), 15),
        )
    )
    assert np.allclose(
        aggregate["spatial_sums"]["all_points"], summary["spatial_sums"]["all_points"]
    )
    assert np.allclose(
        aggregate["pair_sums"]["all_pairs"], summary["pair_sums"]["all_pairs"]
    )
    primitive_residuals = signed_reconstruction_residuals(
        summary["spatial_sums"]["all_points"],
        summary["spatial_sums"]["all_points"],
    )
    pair_residuals = signed_pair_reconstruction_residuals(
        summary["pair_sums"]["all_pairs"], summary["pair_sums"]["all_pairs"]
    )
    assert max(primitive_residuals.values()) == 0.0
    assert max(pair_residuals.values()) == 0.0
