"""Stable numerical and dependency guards for the isolated fixed reference chain."""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np

from lno327.casimir.fixed_outer_q import (
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
)
from lno327.constants import KB


def test_production_source_tree_does_not_import_top_level_validation() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "lno327"
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "validation" or alias.name.startswith("validation."):
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "validation" or module.startswith("validation."):
                    violations.append(f"{path}:{node.lineno}:{module}")
    assert violations == []


def test_fixed_reference_plan_preserves_nested_cutoff_nodes() -> None:
    plan = build_staged_grid_plan(
        u_max_values=(6.0, 10.0, 14.0, 18.0, 24.0),
        radial_orders=(4, 6, 8),
        angular_orders=(4, 8),
        angular_offsets=(0.0, 0.5),
    )
    assert plan.reference_spec_id == "u24_p5_r8_a8_o0p5"
    manifest = build_union_node_manifest(
        plan,
        separation_m=20e-9,
        lattice_a_x_m=3.87e-10,
        lattice_a_y_m=3.87e-10,
    )
    assert len(manifest.labels) == 1040
    assert manifest.grids[plan.reference_spec_id].node_count == 320
    cutoff = plan.ladders["cutoff"]
    for left_id, right_id in zip(cutoff[:-1], cutoff[1:], strict=True):
        left = manifest.labels_by_spec[left_id]
        right = manifest.labels_by_spec[right_id]
        assert right[: len(left)] == left


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {
            "status": "established",
            "working_N": 192,
            "audit_N": 256,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": [
            {
                "N": 256,
                "two_plate_logdet_cross_shift": {"passed": True},
                "shifts": {
                    "shift_0": {
                        "two_plate_logdet": value,
                        "hard_physical_passed": True,
                    }
                },
            }
        ],
    }


def test_fixed_reference_reduction_applies_measure_and_zero_weight_once() -> None:
    plan = build_staged_grid_plan(
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    manifest = build_union_node_manifest(
        plan,
        separation_m=1.0,
        lattice_a_x_m=1.0,
        lattice_a_y_m=1.0,
    )
    points = []
    for label in manifest.labels:
        points.extend((_point("spm", label, 0, 2.0), _point("spm", label, 1, 3.0)))
    results, unresolved = aggregate_certified_outer_q(
        sweet_spot_payload={"point_results": points},
        plan=plan,
        manifest=manifest,
        pairings=("spm",),
        matsubara_indices=(0, 1),
        temperature_K=10.0,
    )
    assert unresolved == []
    result = results[plan.reference_spec_id]["pairings"]["spm"]
    exact_measure = 2.0**2 / (16.0 * np.pi)
    expected = KB * 10.0 * exact_measure * np.asarray([1.0, 3.0])
    np.testing.assert_allclose(result["contributions_J_m2"], expected, rtol=1e-14)
    np.testing.assert_allclose(
        result["partial_free_energy_J_m2"],
        float(np.sum(expected)),
        rtol=1e-14,
    )
    assert result["minimum_working_N"] == 192
    assert result["maximum_audit_N"] == 256
