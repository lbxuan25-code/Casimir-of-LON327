from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "scripts" / "numerical_stability"))

from benchmark_normal_fs_sensitive_sampling import (  # noqa: E402
    REQUIRED_NPZ_FIELDS,
    benchmark_normal_fs_sensitive_sampling,
)


def _small_data():
    return benchmark_normal_fs_sensitive_sampling(
        nk_list=[6, 8],
        eta_list=[1e-3],
        matsubara_list=[1],
        temperature_K=30.0,
        shift_grid_list=[1, 2],
        sampling_modes=["uniform", "multishift_average", "fs_window_refined"],
    )


def test_sampling_modes_run():
    data = _small_data()

    assert set(data["sampling"]) == {"uniform", "multishift_average", "fs_window_refined"}
    assert np.any(data["shift_grid"] == 2)


def test_output_fields_complete():
    data = _small_data()

    assert REQUIRED_NPZ_FIELDS.issubset(data.keys())


def test_response_and_shift_std_are_finite():
    data = _small_data()

    for field in ("sigma_xx", "sigma_yy", "sigma_xy", "sigma_yx"):
        assert np.all(np.isfinite(data[field]))
    assert np.all(np.isfinite(data["sigma_xx_std_over_shifts"]))
    assert np.all(np.isfinite(data["relative_std_over_shifts"]))


def test_fs_diagnostics_are_finite():
    data = _small_data()

    for field in (
        "min_abs_band_energy_on_mesh",
        "points_within_eta",
        "points_within_omega",
        "points_within_kBT",
        "fermi_window_weight_sum",
        "estimated_mesh_energy_resolution",
        "num_refined_points",
    ):
        assert np.all(np.isfinite(data[field]))


def test_symmetry_diagnostics_are_small():
    data = _small_data()

    assert np.max(np.abs(data["delta"])) < 1e-8
    assert np.max(data["relative_offdiag"]) < 1e-8
    assert np.max(data["relative_eigen_split"]) < 1e-8
