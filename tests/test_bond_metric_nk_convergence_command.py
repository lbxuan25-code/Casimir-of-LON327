from __future__ import annotations

import json
from pathlib import Path
import sys

from validation.commands.static.bond_metric_nk_convergence import main


def test_bond_metric_nk_convergence_command_writes_outputs(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "bond_metric_nk.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m validation static bond-metric-nk-convergence",
            "--nks",
            "4",
            "6",
            "--workers",
            "1",
            "--qx",
            "0.03",
            "--qy",
            "0.02",
            "--primitive-tolerance",
            "1",
            "--amplitude-tolerance",
            "1",
            "--phase-tolerance",
            "1",
            "--effective-direct-tolerance",
            "1",
            "--effective-residual-tolerance",
            "1",
            "--longitudinal-tolerance",
            "1",
            "--condition-max",
            "1e16",
            "--observable-relative-tolerance",
            "1",
            "--output",
            str(output),
        ],
    )

    main()

    assert output.is_file()
    assert output.with_suffix(".summary.txt").is_file()
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["schema"] == "dwave_bond_metric_fixed_q_nk_convergence_v1"
    assert [row["nk"] for row in payload["rows"]] == [4, 6]
    for row in payload["rows"]:
        assert row["phase_hessian_policy"] == "nearest_neighbor_bond_metric"
        assert row["valid_for_casimir_input"] is False
        assert "strict_gate_passed" in row
        assert "integer_shift_error_norm" in row
