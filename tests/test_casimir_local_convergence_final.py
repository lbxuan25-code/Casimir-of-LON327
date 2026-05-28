import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_casimir_local_convergence_final.py"
SUMMARY = ROOT / "outputs" / "casimir" / "local_response_integral" / "final_convergence" / "final_convergence_summary.md"
COMMAND = ROOT / "outputs" / "casimir" / "local_response_integral" / "final_convergence" / "final_convergence_command.sh"


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    output_prefix = tmp_path_factory.mktemp("final_convergence") / "quick_final_local_convergence"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_prefix, result


def test_quick_runs(quick_run):
    _output_prefix, result = quick_run

    assert "not a final Casimir conclusion" in result.stdout


def test_dry_run_outputs_full_command():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    command = result.stdout
    assert "scripts/run_casimir_local_convergence_final.py" in command
    assert "--matsubara-max-list 4 8 16 24" in command
    assert "--kparallel-num-list 32 64 96" in command
    assert "--kparallel-max-factor-list 20 40 60" in command
    assert "--phi-num-list 32 64 96" in command
    assert "--normal-nk 96" in command
    assert "--bdg-nk 32" in command


def test_summary_and_command_are_generated(quick_run):
    _output_prefix, _result = quick_run

    assert SUMMARY.exists()
    assert COMMAND.exists()
    summary = SUMMARY.read_text(encoding="utf-8")
    assert "quick_test_result = True" in summary
    assert "full_run_completed = False" in summary
    assert "full_run_pending_user_terminal=True" in summary
    assert "no_full_convergence_conclusion=True" in summary
    assert "not final Casimir conclusion" in summary


def test_outputs_and_flags_are_correct(quick_run):
    output_prefix, _result = quick_run

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert output_prefix.with_suffix(".csv").exists()
        assert np.all(data["local_response"])
        assert not np.any(data["finite_q_resolved"])
        assert np.all(data["benchmark_only"])
        assert np.all(data["not_final_casimir_conclusion"])
        assert set(data["n0_policy"]) == {"skip"}


def test_isotropic_baseline_torque_is_small(quick_run):
    output_prefix, _result = quick_run

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert np.nanmax(data["max_abs_torque_over_theta"]) < 1e-20
        assert all("zero_torque_baseline" in str(item) for item in data["diagnosis"])
