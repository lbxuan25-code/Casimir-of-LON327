from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "numerical_stability"
    / "diagnose_normal_sampling_convergence.py"
)
SPEC = spec_from_file_location("diagnose_normal_sampling_convergence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sampling_script = module_from_spec(SPEC)
SPEC.loader.exec_module(sampling_script)


def _small_sampling(modes=None):
    return sampling_script.diagnose_normal_sampling_convergence(
        nk_list=[6, 8],
        eta_list=[1e-3],
        matsubara_list=[1],
        temperature_K=30.0,
        sampling_modes=["uniform", "shifted", "average"] if modes is None else modes,
    )


def test_sampling_modes_run_and_save_required_fields(tmp_path):
    data = _small_sampling()
    paths = sampling_script.save_outputs(data, tmp_path / "normal_sampling_convergence")

    assert set(data["sampling"]) == {"uniform", "shifted", "average"}
    assert sampling_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert sampling_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_each_sampling_mode_can_run_individually():
    for mode in ("uniform", "shifted", "average"):
        data = _small_sampling([mode])
        assert set(data["sampling"]) == {mode}


def test_normal_sampling_response_is_finite():
    data = _small_sampling()

    for field in ("sigma_xx", "sigma_yy", "sigma_xy", "sigma_yx"):
        assert np.isfinite(data[field]).all()


def test_fs_diagnostics_are_finite():
    data = _small_sampling()

    for field in (
        "min_abs_band_energy_on_mesh",
        "points_within_eta",
        "points_within_omega",
        "points_within_kBT",
        "fermi_window_weight_sum",
        "estimated_mesh_energy_resolution",
    ):
        assert np.isfinite(data[field]).all()


def test_sampling_symmetry_diagnostics_are_small():
    data = _small_sampling()

    assert np.nanmax(np.abs(data["delta"])) < 1e-10
    assert np.nanmax(data["relative_offdiag"]) < 1e-10
    assert np.nanmax(data["relative_eigen_split"]) < 1e-10
