from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "numerical_stability"
    / "convergence_response_imag.py"
)
SPEC = spec_from_file_location("convergence_response_imag", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
convergence_script = module_from_spec(SPEC)
SPEC.loader.exec_module(convergence_script)


def _small_convergence():
    return convergence_script.convergence_response_imag(
        kinds=["normal", "spm", "dwave"],
        nk_list=[4, 6],
        eta_list=[1e-3],
        matsubara_list=[1, 2],
        temperature_K=30.0,
        delta0_eV=0.04,
    )


def test_convergence_script_runs_and_saves_required_fields(tmp_path):
    data = _small_convergence()
    paths = convergence_script.save_outputs(data, tmp_path / "convergence_imag")

    assert data["kind"].size == 12
    assert convergence_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert convergence_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_normal_spm_dwave_responses_are_finite():
    data = _small_convergence()

    for field in (
        "response_xx",
        "response_yy",
        "response_xy",
        "response_yx",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
    ):
        assert np.isfinite(data[field]).all()


def test_symmetry_diagnostics_are_small():
    data = _small_convergence()

    assert np.nanmax(np.abs(data["delta"])) < 1e-10
    assert np.nanmax(data["relative_offdiag"]) < 1e-10
    assert np.nanmax(data["relative_eigen_split"]) < 1e-10


def test_spm_dwave_difference_fields_exist_and_are_finite():
    data = _small_convergence()
    paired = np.isin(data["kind"], ["spm", "dwave"])

    assert np.isfinite(data["spm_dwave_abs_diff_xx"][paired]).all()
    assert np.isfinite(data["spm_dwave_rel_diff_xx"][paired]).all()


def test_convergence_status_and_diagnosis_fields_exist():
    data = _small_convergence()

    assert data["convergence_status"].shape == data["kind"].shape
    assert data["diagnosis"].shape == data["kind"].shape
    assert all(str(item) for item in data["convergence_status"])
    assert all(str(item) for item in data["diagnosis"])
