from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.lib.dwave_global_extrapolation import (
    local_lt_kernel_proxies,
    small_xi_fits,
    static_power_law_fits,
    summarize_fit_ensemble,
)
from validation.run_dwave_small_xi_extrapolation_scan import _fit_channel
from validation.run_dwave_static_global_reference_scan import _run_task


def test_static_power_law_extrapolation_recovers_known_limit():
    nks = np.asarray([64, 80, 96, 112, 144], dtype=int)
    limit = 0.451234
    values = limit + 3.5 / nks.astype(float) ** 2
    fits = static_power_law_fits(
        nks, values, powers=(1, 2, 3), tail_sizes=(3, 4, 5)
    )
    exact = [row for row in fits if row["model"] == "nk^-2"]
    assert exact
    assert min(abs(float(row["intercept"]) - limit) for row in exact) < 1e-11
    summary = summarize_fit_ensemble(fits)
    assert np.isfinite(summary.estimate)
    assert summary.minimum <= limit <= summary.maximum


def test_small_xi_even_fit_recovers_zero_frequency_intercept():
    xi = np.asarray([2e-4, 4e-4, 8e-4, 1.5e-3, 3e-3, 5e-3], dtype=float)
    limit = 0.58321
    values = limit + 7.0 * xi**2 - 25.0 * xi**4
    fits = small_xi_fits(xi, values, tail_sizes=(4, 5, 6))
    even = [row for row in fits if row["model"] == "even_xi2_xi4"]
    assert even
    assert min(abs(float(row["intercept"]) - limit) for row in even) < 1e-10
    summary = summarize_fit_ensemble(fits)
    assert np.isfinite(summary.estimate)
    assert summary.num_accepted_models >= 1


def test_small_xi_runner_fit_channel_exposes_reference_residuals():
    xi = np.asarray([1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3])
    limit = 0.4498
    values = (limit + 4.0 * xi**2).tolist()
    fits, summary = _fit_channel(
        xi,
        values,
        "chi_bar_proxy",
        (4, 5, 6),
        same_grid_static=limit,
        external_reference=limit,
    )
    assert fits
    assert abs(float(summary["estimate"]) - limit) < 1e-8
    assert float(summary["relative_to_same_grid_static"]) < 1e-7
    assert float(summary["relative_to_external_reference"]) < 1e-7


def test_local_lt_kernel_proxies_match_transverse_projection():
    q = np.asarray([3.0, 4.0], dtype=float)
    # Local longitudinal unit vector is (3/5,4/5); transverse is (-4/5,3/5).
    spatial = np.asarray([[2.0, 0.0], [0.0, 5.0]], dtype=complex)
    matrix = np.zeros((3, 3), dtype=complex)
    matrix[0, 0] = -0.4
    matrix[1:3, 1:3] = spatial
    proxies = local_lt_kernel_proxies(SimpleNamespace(k_eff=matrix), q)
    transverse = np.asarray([-4.0 / 5.0, 3.0 / 5.0])
    expected_tt = float(transverse @ spatial.real @ transverse)
    assert np.isclose(proxies["chi_bar_proxy"], 0.4)
    assert np.isclose(proxies["dbar_t_proxy"], -expected_tt)


def test_complete_periodic_static_reference_worker_smoke():
    row = _run_task(
        {
            "nk": 4,
            "shift": (0.5, 0.5),
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "ward_tolerance": 1e-7,
            "ward_absolute_tolerance": 1e-12,
            "condition_max": 1e12,
            "raw_longitudinal_ceiling": 1e-3,
            "longitudinal_tolerance": 1e-7,
            "mixing_tolerance": 1e-7,
            "reality_tolerance": 1e-9,
            "passivity_tolerance": 1e-10,
            "separation_nm": 20.0,
        }
    )
    assert row["nk"] == 4
    assert row["num_k_points"] == 16
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert np.isfinite(row["ward_primitive_mixed_ratio_max"])
    assert row["schur_inverse_method"] == "inv"
