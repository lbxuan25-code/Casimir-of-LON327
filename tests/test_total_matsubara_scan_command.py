from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _THREAD_VARIABLES:
        environment[name] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MKL_DYNAMIC"] = "FALSE"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_environment(),
        check=False,
    )
    assert result.returncode == 0, (
        f"command failed with return code {result.returncode}:\n"
        f"{' '.join(command)}\n\n{result.stdout}"
    )
    return result


def test_total_matsubara_preflight_allows_matching_scan(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    output_root = tmp_path / "scan"

    _run(
        [
            sys.executable,
            "-m",
            "validation",
            "matsubara",
            "orbit-gauss-preflight",
            "--pairings",
            "spm",
            "dwave",
            "--nk",
            "4",
            "--mx",
            "1",
            "--my",
            "0",
            "--matsubara-indices",
            "0",
            "1",
            "--transverse-order",
            "4",
            "--panel-count",
            "1",
            "--transverse-workers",
            "2",
            "--transverse-task-size",
            "2",
            "--legacy-static-nk",
            "4",
            "--legacy-static-order",
            "4",
            "--minimum-speedup",
            "0",
            "--minimum-parallel-cpu-wall-ratio",
            "0",
            "--comparison-rtol",
            "1e-8",
            "--comparison-atol",
            "1e-9",
            "--no-require-physical",
            "--output",
            str(preflight),
        ]
    )
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    assert preflight_payload["status"]["passed"] is True
    assert preflight_payload["status"]["optimization_passed"] is True
    assert preflight_payload["status"]["combined_zero_positive_batch_verified"] is True

    _run(
        [
            sys.executable,
            "-m",
            "validation",
            "matsubara",
            "total-orbit-gauss-scan",
            "--case",
            "smoke:1:0",
            "--pairings",
            "spm",
            "--nk",
            "4",
            "--matsubara-indices",
            "0",
            "1",
            "--gauss-orders",
            "4",
            "8",
            "--panel-count",
            "1",
            "--transverse-workers",
            "2",
            "--transverse-task-size",
            "2",
            "--preflight-manifest",
            str(preflight),
            "--output-root",
            str(output_root),
        ]
    )
    payload = json.loads(
        (output_root / "scan_summary.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "total_matsubara_pointwise_gauss_scan_v2"
    assert payload["status"]["zero_matsubara_included"] is True
    assert payload["status"]["zero_uses_exact_static_divided_difference"] is True
    assert payload["status"]["zero_conductivity_division_used"] is False
    assert payload["status"]["static_physics_contract_softened"] is False
    assert payload["status"]["static_numerical_soft_acceptance_enabled"] is True
    assert payload["status"]["preflight_passed"] is True
    assert payload["preflight"]["accepted"] is True
    assert payload["status"]["valid_for_casimir_input"] is False
