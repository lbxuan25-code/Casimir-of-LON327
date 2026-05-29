import subprocess
import sys
from pathlib import Path

import numpy as np

from lno327.finite_q_response import (
    compare_subspace_and_eigenstate_overlap,
    finite_q_subspace_consistency_diagnostic,
    group_near_degenerate_levels,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_finite_q_subspace_denominator_repair.py"
SUMMARY = ROOT / "outputs" / "response" / "finite_q_subspace_repair" / "finite_q_subspace_repair_summary.md"


def test_subspace_grouping_runs():
    groups = group_near_degenerate_levels(np.array([0.0, 1e-9, 1.0, 1.0 + 1e-9]), deg_tol=1e-7)
    assert [group.size for group in groups] == [2, 2]


def test_projector_and_denominator_fields_are_finite():
    result = finite_q_subspace_consistency_diagnostic(
        "dwave",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=1e-4,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
        deg_tol=1e-7,
        denominator_mode="stable",
    )

    assert np.isfinite(result.projector_overlap_error)
    assert np.isfinite(result.projector_trace_defect)
    assert np.isfinite(result.eigenstate_overlap_offdiag_norm)
    assert np.isfinite(result.denominator_regularization_delta)
    assert result.denominator_mode == "stable"
    assert result.gauge_status == "prototype_not_ward_verified"
    assert not result.final_casimir_input
    assert result.not_final_Casimir_conclusion


def test_raw_and_stable_modes_run_without_skipping_stable_near_degenerate_terms():
    rows = compare_subspace_and_eigenstate_overlap(
        kinds=["normal"],
        matsubara_list=[1],
        q_list=[1e-4],
        q_phi_list=[0.0],
        nk_list=[6],
        deg_tol_list=[1e-7],
        denominator_mode_list=["raw", "stable"],
        temperature_K=30.0,
        delta0=0.04,
        eta=1e-4,
    )

    modes = {row.denominator_mode for row in rows}
    assert modes == {"raw", "stable"}
    stable = [row for row in rows if row.denominator_mode == "stable"][0]
    assert stable.near_degenerate_count >= 0
    assert np.isfinite(stable.near_degenerate_weight_stable)


def test_quick_script_outputs_fields(tmp_path):
    output_prefix = tmp_path / "finite_q_subspace_repair"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    required = {
        "kind",
        "deg_tol",
        "denominator_mode",
        "projector_overlap_error",
        "projector_trace_defect",
        "eigenstate_overlap_offdiag_norm",
        "denominator_regularization_delta",
        "near_degenerate_weight_raw",
        "near_degenerate_weight_stable",
        "stable_denominator_changed_response_norm",
        "stable_denominator_improves_continuity",
        "small_q_relative_error",
        "gauge_status",
        "final_casimir_input",
        "not_final_Casimir_conclusion",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)
        assert set(data["kind"]) == {"normal", "spm", "dwave"}
        assert set(data["denominator_mode"]) == {"raw", "stable"}
        assert np.all(np.isfinite(data["projector_overlap_error"]))
        assert np.all(np.isfinite(data["denominator_regularization_delta"]))
        assert set(data["gauge_status"]) == {"prototype_not_ward_verified"}
        assert not np.any(data["final_casimir_input"])
        assert np.all(data["not_final_Casimir_conclusion"])
    assert SUMMARY.exists()
