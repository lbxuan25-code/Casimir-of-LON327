import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "casimir" / "refine_casimir_local_convergence_blockers.py"
SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "archive"
    / "casimir"
    / "local_response_integral"
    / "refined_convergence"
    / "refined_convergence_summary.md"
)
COMMAND = (
    ROOT
    / "validation"
    / "outputs"
    / "archive"
    / "casimir"
    / "local_response_integral"
    / "refined_convergence"
    / "refined_convergence_command.sh"
)


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    output_prefix = tmp_path_factory.mktemp("refined_convergence") / "quick_refined_local_convergence"
    cache_dir = tmp_path_factory.mktemp("response_cache")
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
    return output_prefix, cache_dir, result


def test_quick_runs(quick_run):
    _output_prefix, _cache_dir, result = quick_run

    assert "refined convergence benchmark only" in result.stdout
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
    assert "validation/scripts/casimir/refine_casimir_local_convergence_blockers.py" in command
    assert "--u-max-list 20 40 60 80" in command
    assert "--du 0.5" in command
    assert "--matsubara-max-list 24 32 48 64" in command
    assert "--energy-theta-list 0" in command
    assert "--torque-check-theta-list 0 0.7853981634 1.5707963268" in command
    assert "--use-response-cache" in command
    assert "--normal-nk 96" in command
    assert "--bdg-nk 32" in command


def test_u_max_du_mapping_is_recorded(quick_run):
    output_prefix, _cache_dir, _result = quick_run

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        mask = data["scan_type"] == "cutoff"
        assert set(data["u_max"][mask]) == {4.0, 6.0}
        for u_max, du, kparallel_num, kparallel_max_factor in zip(
            data["u_max"][mask],
            data["du"][mask],
            data["implied_kparallel_num"][mask],
            data["kparallel_max_factor"][mask],
        ):
            assert du == 2.0
            assert kparallel_num == int(u_max / du) + 1
            assert kparallel_max_factor == u_max


def test_summary_and_command_are_generated(quick_run):
    output_prefix, _cache_dir, _result = quick_run

    summary_path = output_prefix.parent / "refined_convergence_summary.md"
    command_path = output_prefix.parent / "refined_convergence_command.sh"
    assert summary_path.exists()
    assert command_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "quick_test_only=True" in summary
    assert "no_full_convergence_conclusion=True" in summary
    assert "full_run_pending_user_terminal=True" in summary
    assert "fixed kparallel_num caused changing du when cutoff increased" in summary
    assert "new_cutoff_scan = u=k_parallel*d with fixed du" in summary
    assert "response_cache_used=True" in summary
    assert "response_cache_rebuilt=True" in summary


def test_outputs_and_flags_are_correct(quick_run):
    output_prefix, _cache_dir, _result = quick_run

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert output_prefix.with_suffix(".csv").exists()
        assert np.all(data["local_response"])
        assert not np.any(data["finite_momentum_resolved"])
        assert np.all(data["benchmark_only"])
        assert np.all(data["not_final_casimir_conclusion"])
        assert set(data["n0_policy"]) == {"skip"}


def test_isotropic_baseline_torque_is_small(quick_run):
    output_prefix, _cache_dir, _result = quick_run

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert np.nanmax(data["max_abs_torque_over_theta"]) < 1e-20
        assert all("zero_torque_baseline" in str(item) for item in data["diagnosis"])


def test_response_cache_file_is_generated(quick_run):
    _output_prefix, cache_dir, _result = quick_run

    assert any(cache_dir.glob("response_tensor_*.npz"))


def test_second_run_reads_response_cache(quick_run, tmp_path):
    _output_prefix, cache_dir, _result = quick_run
    second_output = tmp_path / "second_cached"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--use-response-cache",
            "--cache-dir",
            str(cache_dir),
            "--output-prefix",
            str(second_output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = SUMMARY.read_text(encoding="utf-8")
    assert "response_cache_used=True" in summary
    assert "response_cache_rebuilt=False" in summary
    assert "response_cache_misses=0" in summary


def test_cache_and_no_cache_small_results_match(tmp_path):
    cached_output = tmp_path / "cached"
    uncached_output = tmp_path / "uncached"
    cache_dir = tmp_path / "cache"
    common = [
        sys.executable,
        str(SCRIPT),
        "--quick",
    ]
    subprocess.run(
        [
            *common,
            "--use-response-cache",
            "--rebuild-response-cache",
            "--cache-dir",
            str(cache_dir),
            "--output-prefix",
            str(cached_output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            *common,
            "--output-prefix",
            str(uncached_output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with np.load(cached_output.with_suffix(".npz"), allow_pickle=True) as cached:
        with np.load(uncached_output.with_suffix(".npz"), allow_pickle=True) as uncached:
            assert np.allclose(cached["energy"], uncached["energy"])
            assert np.allclose(cached["max_abs_torque_over_theta"], uncached["max_abs_torque_over_theta"])
            assert np.all(cached["local_response"])
            assert not np.any(cached["finite_momentum_resolved"])
            assert set(cached["n0_policy"]) == {"skip"}
