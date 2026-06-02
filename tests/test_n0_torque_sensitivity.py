from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "numerical_stability"
    / "assess_n0_torque_sensitivity.py"
)
SPEC = spec_from_file_location("assess_n0_torque_sensitivity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
n0_script = module_from_spec(SPEC)
SPEC.loader.exec_module(n0_script)


def _small_assessment(include_toy=False):
    return n0_script.assess_n0_sensitivity(
        kinds=["normal", "spm", "dwave"],
        nk=4,
        temperature_K=30.0,
        delta0_eV=0.04,
        eta_eV=1e-4,
        distance_m=3e-8,
        k_parallel=1e6,
        phi=0.2,
        theta=0.7,
        reference_min=1,
        reference_max=2,
        sensitivity_threshold=0.01,
        theta_scan_min=0.0,
        theta_scan_max=np.pi,
        theta_scan_num=9,
        include_toy_anisotropic_control=include_toy,
    )


def _row(data, kind, policy):
    mask = (data["kind"] == kind) & (data["policy"] == policy)
    assert np.count_nonzero(mask) == 1
    return int(np.flatnonzero(mask)[0])


def test_required_fields_are_saved(tmp_path):
    data = _small_assessment()
    paths = n0_script.save_outputs(data, tmp_path / "n0_sensitivity")

    assert n0_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert n0_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_current_local_baseline_n0_proxy_has_no_significant_anisotropy():
    data = _small_assessment()

    for kind in ("normal", "spm", "dwave"):
        index = _row(data, kind, "extrapolate_from_lowest_matsubara")
        assert abs(data["delta_n0_proxy"][index]) < 1e-10
        assert data["relative_offdiag_n0_proxy"][index] < 1e-10


def test_static_kernel_does_not_enter_reflection_integrand():
    data = _small_assessment()

    for kind in ("normal", "spm", "dwave"):
        index = _row(data, kind, "use_static_kernel")
        assert data["not_used_as_sigma"][index]
        assert np.isnan(data["tau_n0_proxy"][index].real)
        assert np.isnan(data["ratio_abs_n0_to_ref"][index])
        assert np.isnan(data["tau_n0_proxy_theta"][index].real).all()
        assert "static kernel is not Sigma_SC(0)" in data["notes"][index]


def test_ratio_has_clear_zero_baseline_status():
    data = _small_assessment()

    for kind in ("normal", "spm", "dwave"):
        index = _row(data, kind, "skip")
        assert np.isfinite(data["ratio_abs_n0_to_ref"][index])
        assert data["ratio_abs_n0_to_ref"][index] == 0.0
        assert data["n0_sensitivity"][index] == "negligible_zero_baseline"
        assert data["skip_acceptability"][index] == "acceptable_for_current_local_baseline"


def test_toy_anisotropic_control_produces_reference_torque_and_ratio():
    data = _small_assessment(include_toy=True)
    index = _row(data, "toy_anisotropic", "skip")

    assert abs(data["tau_ref_n_ge1"][index]) > 0.0
    assert np.isfinite(data["ratio_abs_n0_to_ref"][index])
    assert data["tau_ref_theta"][index].shape == data["theta_scan"].shape


def test_skip_acceptability_classification_logic():
    negligible = n0_script._classify_sensitivity(0.0 + 0.0j, 0.0 + 0.0j, 0.0, 0.01)
    small = n0_script._classify_sensitivity(1.0 + 0.0j, 0.001 + 0.0j, 0.001, 0.01)
    large = n0_script._classify_sensitivity(1.0 + 0.0j, 0.2 + 0.0j, 0.2, 0.01)
    zero_ref = n0_script._classify_sensitivity(0.0 + 0.0j, 1e-6 + 0.0j, 1e294, 0.01)

    assert negligible == ("negligible_zero_baseline", "acceptable_for_current_local_baseline")
    assert small == ("below_threshold", "acceptable_for_current_local_baseline")
    assert large == ("above_threshold", "not_acceptable_requires_zero_frequency_model")
    assert zero_ref == (
        "finite_n0_proxy_over_zero_reference",
        "not_acceptable_requires_zero_frequency_model",
    )
