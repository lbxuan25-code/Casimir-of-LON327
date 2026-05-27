from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_normal_fs_adaptive_integration import (  # noqa: E402
    REQUIRED_NPZ_FIELDS,
    benchmark_normal_fs_adaptive_integration,
    fs_adaptive_mesh,
)
from lno327.normal_sampling import normal_sheet_tensor_from_sampling  # noqa: E402


def _small_data():
    return benchmark_normal_fs_adaptive_integration(
        nk_list=[6, 8],
        eta_list=[1e-3],
        matsubara_list=[1],
        temperature_K=30.0,
        refine_factor_list=[2],
        fs_window_factor=1.0,
        sampling_modes=["uniform", "multishift_average", "fs_adaptive"],
        shift_grid=2,
    )


def test_sampling_modes_run():
    data = _small_data()

    assert set(data["sampling"]) == {"uniform", "multishift_average", "fs_adaptive"}


def test_output_fields_complete():
    data = _small_data()

    assert REQUIRED_NPZ_FIELDS.issubset(data.keys())


def test_adaptive_weights_are_normalized():
    _mesh, weights, metadata = fs_adaptive_mesh(
        nk=6,
        eta_eV=1e-3,
        omega_eV=0.02,
        temperature_K=30.0,
        refine_factor=2,
    )

    assert np.isclose(np.sum(weights), 1.0)
    assert np.isclose(metadata["weight_sum"], 1.0)


def test_response_and_fs_diagnostics_are_finite():
    data = _small_data()

    for field in (
        "sigma_xx",
        "sigma_yy",
        "sigma_xy",
        "sigma_yx",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
        "min_abs_band_energy_on_mesh",
        "fermi_window_weight_sum",
        "estimated_mesh_energy_resolution",
    ):
        assert np.all(np.isfinite(data[field]))


def test_symmetry_diagnostics_are_small():
    data = _small_data()

    assert np.max(np.abs(data["delta"])) < 1e-8
    assert np.max(data["relative_offdiag"]) < 1e-8
    assert np.max(data["relative_eigen_split"]) < 1e-8


def test_adaptive_has_at_least_uniform_kpoints():
    data = _small_data()

    for nk in [6, 8]:
        uniform = data["num_kpoints_total"][(data["sampling"] == "uniform") & (data["nk"] == nk)][0]
        adaptive = data["num_kpoints_total"][(data["sampling"] == "fs_adaptive") & (data["nk"] == nk)][0]
        assert adaptive >= uniform


def test_normal_sampling_module_shared_sheet_tensor():
    tensor = normal_sheet_tensor_from_sampling(
        omega_eV=0.02,
        temperature_K=30.0,
        eta_eV=1e-3,
        nk=6,
        sampling="fs_adaptive",
        refine_factor=2,
    )

    assert np.isfinite(tensor.xx)
    assert np.isfinite(tensor.yy)
