from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from lno327 import PairingAmplitudes, local_response_imag_axis

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "scripts"
    / "response"
    / "compare_static_response_policies.py"
)
SPEC = spec_from_file_location("compare_static_response_policies", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_script = module_from_spec(SPEC)
SPEC.loader.exec_module(compare_script)


def _small_comparison(kinds=None, policies=None):
    return compare_script.compare_static_policies(
        kinds=["normal", "spm", "dwave"] if kinds is None else kinds,
        policies=list(compare_script.POLICIES) if policies is None else policies,
        nk=4,
        temperature_K=30.0,
        delta0_eV=0.04,
        eta_eV=1e-4,
        distance_m=3e-8,
        k_parallel=1e6,
        phi=0.2,
        theta=0.7,
    )


def _row(data, kind, policy):
    mask = (data["kind"] == kind) & (data["policy"] == policy)
    assert np.count_nonzero(mask) == 1
    return int(np.flatnonzero(mask)[0])


def test_compare_static_policy_outputs_required_fields(tmp_path):
    data = _small_comparison(kinds=["spm"], policies=["skip"])
    paths = compare_script.save_outputs(data, tmp_path / "static_policy_comparison")

    assert compare_script.REQUIRED_NPZ_FIELDS.issubset(data)
    with np.load(paths[0], allow_pickle=True) as loaded:
        assert compare_script.REQUIRED_NPZ_FIELDS.issubset(loaded.files)


def test_skip_policy_produces_no_matrix_or_integrand():
    data = _small_comparison(kinds=["spm"], policies=["skip"])
    index = _row(data, "spm", "skip")

    assert data["status"][index] == "skipped"
    assert not data["matrix_finite"][index]
    assert np.isnan(data["response_xx"][index].real)
    assert np.isnan(data["energy_integrand_n0"][index].real)
    assert np.isnan(data["torque_integrand_n0"][index].real)
    assert "n=0 omitted in current local baseline" in data["notes"][index]


def test_extrapolate_policy_is_approximate_and_has_finite_diagnostic_integrand():
    data = _small_comparison(kinds=["dwave"], policies=["extrapolate_from_lowest_matsubara"])
    index = _row(data, "dwave", "extrapolate_from_lowest_matsubara")

    assert data["status"][index] == "extrapolated"
    assert data["approximate"][index]
    assert data["matrix_finite"][index]
    assert not data["not_used_as_sigma"][index]
    assert np.isfinite(data["sheet_conductivity_xx"][index])
    assert np.isfinite(data["reflection_dimensionless_xx"][index])
    assert np.isfinite(data["energy_integrand_n0"][index])
    assert np.isfinite(data["torque_integrand_n0"][index])
    assert "sensitivity estimate only, not final n=0 physics" in data["notes"][index]


def test_static_kernel_policy_is_not_used_as_sigma_or_reflection_input():
    data = _small_comparison(kinds=["spm", "dwave"], policies=["use_static_kernel"])

    for kind in ("spm", "dwave"):
        index = _row(data, kind, "use_static_kernel")
        assert data["status"][index] == "static_kernel"
        assert data["matrix_finite"][index]
        assert data["not_used_as_sigma"][index]
        assert np.isnan(data["sheet_conductivity_xx"][index].real)
        assert np.isnan(data["reflection_dimensionless_xx"][index].real)
        assert np.isnan(data["energy_integrand_n0"][index].real)
        assert np.isnan(data["torque_integrand_n0"][index].real)
        assert "static kernel is not Sigma_SC(0)" in data["notes"][index]


def test_bdg_zero_frequency_sigma_sc_division_remains_forbidden():
    with pytest.raises(ValueError, match="n=0 is unresolved"):
        local_response_imag_axis(
            "spm",
            0.0,
            [(0.0, 0.0)],
            temperature_K=30.0,
            pairing_params=PairingAmplitudes(delta0_eV=0.04),
        )


def test_current_local_baseline_does_not_generate_artificial_anisotropy():
    data = _small_comparison()
    finite = data["matrix_finite"]

    assert np.nanmax(np.abs(data["delta"][finite])) < 1e-10
    assert np.nanmax(data["relative_offdiag"][finite]) < 1e-10
    assert np.nanmax(data["relative_eigen_split"][finite]) < 1e-10
