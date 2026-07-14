from __future__ import annotations

import csv
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


def test_dwave_integrand_profile_writes_node_boundary_panel_and_npz_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation",
            "diagnostic",
            "dwave-orbit-integrand-profile",
            "--nk",
            "4",
            "--mx",
            "1",
            "--my",
            "1",
            "--matsubara-indices",
            "0",
            "1",
            "--gauss-order",
            "8",
            "--panel-count",
            "2",
            "--boundary-epsilon",
            "1e-6",
            "--transverse-workers",
            "2",
            "--transverse-task-size",
            "2",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_environment(),
        check=False,
    )
    assert result.returncode == 0, result.stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "dwave_complete_orbit_transverse_integrand_profile_v1"
    assert payload["status"]["same_batched_complete_orbit_evaluator"] is True
    assert payload["status"]["primitive_integrated_before_postprocessing"] is True
    assert payload["status"]["diagnostic_only"] is True
    assert payload["status"]["valid_for_casimir_input"] is False
    assert payload["config"]["gauss_order"] == 8
    assert payload["config"]["panel_count"] == 2
    assert payload["evaluator_profile"]["material_workspace_implementation"] == (
        "batched_model_capability"
    )
    assert payload["evaluator_profile"]["q_workspace_implementation"] == (
        "batched_model_capability"
    )
    assert len(payload["full_integral_physical_rows"]) == 2
    assert {row["matsubara_index"] for row in payload["full_integral_physical_rows"]} == {
        0,
        1,
    }

    nodes = Path(payload["outputs"]["nodes"])
    boundaries = Path(payload["outputs"]["boundaries"])
    panels = Path(payload["outputs"]["panels"])
    npz = Path(payload["outputs"]["npz"])
    summary = Path(payload["outputs"]["summary"])
    for path in (nodes, boundaries, panels, npz, summary):
        assert path.is_file(), path

    with nodes.open(newline="", encoding="utf-8") as handle:
        node_rows = list(csv.DictReader(handle))
    with boundaries.open(newline="", encoding="utf-8") as handle:
        boundary_rows = list(csv.DictReader(handle))
    with panels.open(newline="", encoding="utf-8") as handle:
        panel_rows = list(csv.DictReader(handle))

    assert len(node_rows) == 8
    assert len(boundary_rows) == 2
    assert len(panel_rows) == 2
    assert all("n0_bubble_next_relative" in row for row in node_rows)
    assert all("n1_bubble_next_relative" in row for row in node_rows)
    assert all("packed_left_right_relative" in row for row in boundary_rows)
    assert all("packed_integral_norm_over_full" in row for row in panel_rows)
