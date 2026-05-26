from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refine_high_nk_convergence.py"
SPEC = spec_from_file_location("refine_high_nk_convergence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
high_nk_script = module_from_spec(SPEC)
SPEC.loader.exec_module(high_nk_script)


def _small_high_nk():
    return high_nk_script.refine_high_nk_convergence(
        kinds=["normal", "spm", "dwave"],
        nk_list=[4, 6, 8],
        eta_list=[1e-3],
        matsubara_list=[1],
        temperature_K=30.0,
        delta0_eV=0.04,
    )


def test_high_nk_script_runs_and_saves_required_fields(tmp_path):
    data = _small_high_nk()
    paths = high_nk_script.save_outputs(data, tmp_path / "high_nk_convergence")

    assert data["kind"].size == 9
    assert high_nk_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert high_nk_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_high_nk_responses_are_finite():
    data = _small_high_nk()

    for field in (
        "response_xx",
        "response_yy",
        "response_xy",
        "response_yx",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
    ):
        assert np.isfinite(data[field]).all()


def test_high_nk_symmetry_diagnostics_are_small():
    data = _small_high_nk()

    assert np.nanmax(np.abs(data["delta"])) < 1e-10
    assert np.nanmax(data["relative_offdiag"]) < 1e-10
    assert np.nanmax(data["relative_eigen_split"]) < 1e-10


def test_last_two_relative_change_field_is_finite():
    data = _small_high_nk()

    assert np.isfinite(data["relative_change_between_last_two_nk"]).all()


def test_status_and_pairing_difference_fields_exist():
    data = _small_high_nk()
    paired = np.isin(data["kind"], ["spm", "dwave"])

    assert all(str(item) for item in data["high_nk_convergence_status"])
    assert np.isfinite(data["spm_dwave_rel_diff_xx"][paired]).all()
