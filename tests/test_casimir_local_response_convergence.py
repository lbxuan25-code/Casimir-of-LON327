from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "casimir"
    / "converge_casimir_local_response_integral.py"
)
SPEC = spec_from_file_location("converge_casimir_local_response_integral", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
conv = module_from_spec(SPEC)
SPEC.loader.exec_module(conv)


def _small_data():
    return conv.converge_casimir_local_response_integral(
        kinds=["normal", "spm", "dwave"],
        distance_m=5e-8,
        theta_list=[0.0, np.pi / 4.0, np.pi / 2.0],
        matsubara_max_list=[1, 2],
        kparallel_num_list=[6, 8],
        kparallel_max_factor_list=[10.0, 20.0],
        phi_num_list=[6, 8],
        temperature_K=30.0,
        normal_nk=12,
        normal_eta_eV=1e-4,
        normal_sampling="fs_adaptive",
        normal_refine_factor=2,
        bdg_nk=8,
        delta0_eV=0.04,
    )


def test_convergence_function_runs():
    data = _small_data()

    assert set(data["kind"]) == {"normal", "spm", "dwave"}
    assert set(data["scan_type"]) == {"matsubara", "kparallel_num", "kparallel_cutoff", "phi"}


def test_output_fields_complete():
    data = _small_data()

    assert conv.REQUIRED_NPZ_FIELDS.issubset(data.keys())


def test_energy_torque_and_indicators_are_finite_or_nan():
    data = _small_data()

    for field in (
        "max_abs_energy_over_theta",
        "max_abs_torque_over_theta",
        "relative_change_vs_largest_setting",
        "last_two_relative_change",
        "matsubara_tail_indicator",
    ):
        assert np.all(np.isfinite(data[field]) | np.isnan(data[field]))


def test_flags_are_correct():
    data = _small_data()

    assert np.all(data["local_response"])
    assert not np.any(data["finite_q_resolved"])
    assert np.all(data["benchmark_only"])
    assert np.all(data["not_final_casimir_conclusion"])
    assert set(data["n0_policy"]) == {"skip"}


def test_isotropic_baseline_has_near_zero_torque():
    data = _small_data()

    assert np.nanmax(data["max_abs_torque_over_theta"]) < 1e-20
    assert all("zero_torque_baseline" in str(item) for item in data["diagnosis"])
