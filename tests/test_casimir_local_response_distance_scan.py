import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "casimir" / "benchmark_casimir_local_response_distance_scan.py"
SCAN_ROOT = ROOT / "validation" / "outputs" / "casimir" / "local_response_integral" / "distance_scan"
SUMMARY = SCAN_ROOT / "distance_scan_summary.md"
COMMAND = SCAN_ROOT / "distance_scan_command.sh"


def _run_quick(tmp_path):
    output_prefix = tmp_path / "distance_scan"
    cache_dir = tmp_path / "cache"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--use-response-cache",
            "--rebuild-response-cache",
            "--cache-dir",
            str(cache_dir),
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_prefix, result


def test_quick_mode_runs(tmp_path):
    output_prefix, result = _run_quick(tmp_path)

    assert output_prefix.with_suffix(".npz").exists()
    assert "local-response distance scan benchmark only" in result.stdout


def test_dry_run_outputs_full_command():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    command = result.stdout
    assert "validation/scripts/casimir/benchmark_casimir_local_response_distance_scan.py" in command
    assert "--distance-list 3e-08 5e-08 7.5e-08 1e-07 1.5e-07 2e-07" in command
    assert "--matsubara-max 64" in command
    assert "--u-max 80" in command
    assert "--du 0.5" in command
    assert "--use-response-cache" in command


def test_command_file_is_generated(tmp_path):
    _run_quick(tmp_path)

    assert COMMAND.exists()


def test_output_fields_complete(tmp_path):
    output_prefix, _result = _run_quick(tmp_path)
    required = {
        "kind",
        "distance_m",
        "theta",
        "temperature_K",
        "matsubara_max",
        "u_max",
        "du",
        "kparallel_num",
        "phi_num",
        "energy",
        "torque_fd",
        "max_abs_torque_over_theta",
        "energy_abs",
        "normal_sampling",
        "normal_nk",
        "normal_refine_factor",
        "bdg_nk",
        "delta0",
        "response_cache_used",
        "n0_policy",
        "local_response",
        "finite_momentum_resolved",
        "benchmark_only",
        "not_final_casimir_conclusion",
        "zero_torque_baseline",
        "warning_possible_spurious_torque",
        "diagnosis",
        "notes",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)


def test_flags_are_correct(tmp_path):
    output_prefix, _result = _run_quick(tmp_path)

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert np.all(data["local_response"])
        assert not np.any(data["finite_momentum_resolved"])
        assert np.all(data["benchmark_only"])
        assert np.all(data["not_final_casimir_conclusion"])
        assert set(data["n0_policy"]) == {"skip"}


def test_quick_isotropic_baseline_is_zero(tmp_path):
    output_prefix, _result = _run_quick(tmp_path)

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        for kind in ["normal", "spm", "dwave"]:
            mask = data["kind"] == kind
            assert np.any(mask)
            assert np.nanmax(np.abs(data["torque_fd"][mask])) < 1e-20
            assert np.all(data["zero_torque_baseline"][mask])


def test_toy_anisotropic_control_is_nonzero(tmp_path):
    output_prefix, _result = _run_quick(tmp_path)

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        mask = data["kind"] == "toy_anisotropic"
        assert np.any(mask)
        assert np.nanmax(np.abs(data["torque_fd"][mask])) > 1e-20
        assert any("plumbing_pass_toy_anisotropy" in str(item) for item in data["diagnosis"][mask])


def test_quick_summary_has_no_distance_scan_conclusion(tmp_path):
    _run_quick(tmp_path)

    summary = SUMMARY.read_text(encoding="utf-8")
    assert "quick_test_only=True" in summary
    assert "no_distance_scan_conclusion=True" in summary
    assert "full_run_pending_user_terminal=True" in summary
    assert "ready_for_anisotropy_mechanism_benchmark=False" in summary
