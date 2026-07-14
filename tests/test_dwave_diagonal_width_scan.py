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


def test_dwave_diagonal_width_scan_writes_cut_comparisons(tmp_path: Path) -> None:
    output = tmp_path / "width.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation",
            "matsubara",
            "dwave-diagonal-width-scan",
            "--nk",
            "4",
            "--directions",
            "1,1",
            "2,1",
            "--matsubara-indices",
            "0",
            "1",
            "--gauss-order",
            "8",
            "--panel-count",
            "2",
            "--integration-starts",
            "-3.141592653589793",
            "-2.945243112740431",
            "--max-point-evaluations",
            "128",
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
    assert payload["schema"] == "dwave_diagonal_width_scan_v1"
    assert payload["config"]["directions"] == [[1, 1], [2, 1]]
    assert payload["config"]["matsubara_indices"] == [0, 1]
    assert payload["config"]["gauss_order"] == 8
    assert len(payload["direction_summaries"]) == 2
    assert len(payload["raw_rows"]) == 8
    assert payload["status"]["diagnostic_only"] is True
    assert payload["status"]["valid_for_casimir_input"] is False

    directions_path = output.with_name("width.directions.csv")
    raw_path = output.with_name("width.raw.csv")
    summary_path = output.with_name("width.summary.txt")
    for path in (directions_path, raw_path, summary_path):
        assert path.is_file(), path

    with directions_path.open(newline="", encoding="utf-8") as handle:
        directions = list(csv.DictReader(handle))
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))

    assert len(directions) == 2
    assert len(raw) == 8
    assert {row["classification"] for row in directions}
    assert {int(row["matsubara_index"]) for row in raw} == {0, 1}
    assert {int(row["cut_index"]) for row in raw} == {0, 1}
