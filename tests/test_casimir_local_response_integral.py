from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from lno327.normal_sampling import normal_sheet_tensor_from_sampling

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_casimir_local_response_integral.py"
SPEC = spec_from_file_location("benchmark_casimir_local_response_integral", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def _small_data(include_toy=False):
    return bench.benchmark_casimir_local_response_integral(
        kinds=["normal", "spm", "dwave"],
        distance_list=[5e-8],
        theta_list=[0.0, np.pi / 4.0, np.pi / 2.0],
        matsubara_min=1,
        matsubara_max=2,
        kparallel_num=8,
        kparallel_max_factor=20.0,
        phi_num=8,
        temperature_K=30.0,
        normal_nk=12,
        normal_eta_eV=1e-4,
        normal_sampling="fs_adaptive",
        normal_refine_factor=2,
        bdg_nk=8,
        delta0_eV=0.04,
        include_toy_anisotropic_control=include_toy,
    )


def test_integral_function_runs():
    data = _small_data()

    assert set(data["kind"]) == {"normal", "spm", "dwave"}


def test_output_fields_complete():
    data = _small_data()

    assert bench.REQUIRED_NPZ_FIELDS.issubset(data.keys())


def test_energy_and_torque_are_finite():
    data = _small_data()

    assert np.all(np.isfinite(data["energy"]))
    assert np.all(np.isfinite(data["torque_fd"]))


def test_flags_are_correct():
    data = _small_data()

    assert np.all(data["local_response"])
    assert not np.any(data["finite_q_resolved"])
    assert np.all(data["benchmark_only"])
    assert np.all(data["not_final_casimir_conclusion"])
    assert set(data["n0_policy"]) == {"skip"}


def test_isotropic_baseline_has_near_zero_torque():
    data = _small_data()

    assert np.nanmax(np.abs(data["torque_fd"])) < 1e-20
    assert all("zero_torque_baseline" in str(item) for item in data["diagnosis"])


def test_toy_anisotropic_control_has_nonzero_torque():
    data = _small_data(include_toy=True)
    mask = data["kind"] == "toy_anisotropic"

    assert np.any(mask)
    assert np.nanmax(np.abs(data["torque_fd"][mask])) > 1e-20
    assert any("plumbing_pass_toy_anisotropy" in str(item) for item in data["diagnosis"][mask])


def test_casimir_benchmark_uses_shared_normal_sampling_module():
    tensor = normal_sheet_tensor_from_sampling(
        omega_eV=0.02,
        temperature_K=30.0,
        eta_eV=1e-3,
        nk=6,
        sampling="uniform",
        refine_factor=2,
    )

    assert np.isfinite(tensor.xx)
