from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_bdg_normal_limit.py"
SPEC = spec_from_file_location("benchmark_bdg_normal_limit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_script = module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_script)


def _small_benchmark():
    return benchmark_script.benchmark_bdg_normal_limit(
        kinds=["spm", "dwave"],
        delta0_list=[0.0, 1e-5, 1e-3],
        nk=4,
        temperature_K=30.0,
        matsubara_index=1,
        eta_eV=1e-4,
    )


def test_benchmark_runs_and_saves_required_fields(tmp_path):
    data = _small_benchmark()
    paths = benchmark_script.save_outputs(data, tmp_path / "bdg_normal_limit")

    assert benchmark_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert benchmark_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_delta0_zero_spm_dwave_sigma_is_finite():
    data = _small_benchmark()
    zero_rows = np.isclose(data["delta0"], 0.0)

    assert np.isfinite(data["Sigma_xx"][zero_rows]).all()
    assert np.isfinite(data["Sigma_yy"][zero_rows]).all()
    assert np.isfinite(data["Sigma_xy"][zero_rows]).all()
    assert np.isfinite(data["Sigma_yx"][zero_rows]).all()


def test_delta0_zero_spm_dwave_responses_coincide():
    data = _small_benchmark()
    spm_zero = (data["kind"] == "spm") & np.isclose(data["delta0"], 0.0)
    dwave_zero = (data["kind"] == "dwave") & np.isclose(data["delta0"], 0.0)

    spm_matrix = np.array(
        [
            [data["Sigma_xx"][spm_zero][0], data["Sigma_xy"][spm_zero][0]],
            [data["Sigma_yx"][spm_zero][0], data["Sigma_yy"][spm_zero][0]],
        ]
    )
    dwave_matrix = np.array(
        [
            [data["Sigma_xx"][dwave_zero][0], data["Sigma_xy"][dwave_zero][0]],
            [data["Sigma_yx"][dwave_zero][0], data["Sigma_yy"][dwave_zero][0]],
        ]
    )

    np.testing.assert_allclose(spm_matrix, dwave_matrix, atol=1e-12)


def test_symmetry_diagnostics_remain_small():
    data = _small_benchmark()

    assert np.nanmax(np.abs(data["delta"])) < 1e-10
    assert np.nanmax(data["relative_offdiag"]) < 1e-10
    assert np.nanmax(data["relative_eigen_split"]) < 1e-10


def test_bdg_to_normal_ratio_is_finite_not_forced_to_one():
    data = _small_benchmark()

    assert np.isfinite(data["ratio_sigma_xx_to_normal"]).all()
    assert data["ratio_sigma_xx_to_normal"].shape == data["Sigma_xx"].shape
