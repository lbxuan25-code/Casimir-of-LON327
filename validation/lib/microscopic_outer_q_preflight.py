"""Composite-panel facade for microscopic outer-q preflight helpers.

The previously qualified aggregation and comparison logic is retained verbatim in
``microscopic_outer_q_preflight_legacy``.  Only outer-grid planning and node
construction are replaced by nested cumulative radial panels.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from validation.lib import microscopic_outer_q_preflight_legacy as _legacy
from validation.lib.microscopic_outer_q_compound import (
    OuterQGridPlan,
    OuterQGridSpec,
    OuterQNodeManifest,
    build_staged_grid_plan,
    build_union_node_manifest,
)

# Re-export the retained helper surface, including private utilities used by tests.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))


def aggregate_certified_outer_q(
    *,
    sweet_spot_payload: Mapping[str, Any],
    plan: OuterQGridPlan,
    manifest: OuterQNodeManifest,
    pairings: Sequence[str],
    matsubara_indices: Sequence[int],
    temperature_K: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results, unresolved = _legacy.aggregate_certified_outer_q(
        sweet_spot_payload=sweet_spot_payload,
        plan=plan,
        manifest=manifest,
        pairings=pairings,
        matsubara_indices=matsubara_indices,
        temperature_K=temperature_K,
    )
    by_id = {spec.spec_id: spec for spec in plan.specs}
    for spec_id, record in results.items():
        spec = by_id[spec_id]
        grid = manifest.grids[spec_id]
        record["spec"].update(
            {
                "radial_rule": "nested_composite_gauss_legendre",
                "radial_panel_edges": list(spec.radial_panel_edges),
                "radial_panel_count": spec.radial_panel_count,
                "radial_panel_order": spec.radial_panel_order,
                "total_radial_nodes": int(grid.radial_order),
                "nested_cutoff_node_reuse": True,
            }
        )
    return results, unresolved


def compare_ladders(
    *,
    plan: OuterQGridPlan,
    config_results: Mapping[str, Any],
    pairings: Sequence[str],
    absolute_tolerance_J_m2: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    output = _legacy.compare_ladders(
        plan=plan,
        config_results=config_results,
        pairings=pairings,
        absolute_tolerance_J_m2=absolute_tolerance_J_m2,
        relative_tolerance=relative_tolerance,
    )
    by_id = {spec.spec_id: spec for spec in plan.specs}
    for pairing in pairings:
        record = output.get("cutoff", {}).get(str(pairing), {})
        for comparison in record.get("comparisons", []):
            left_id = comparison.get("left_spec_id")
            right_id = comparison.get("right_spec_id")
            if left_id not in by_id or right_id not in by_id:
                continue
            left_spec = by_id[left_id]
            right_spec = by_id[right_id]
            nested = (
                right_spec.radial_panel_edges[:-1]
                == left_spec.radial_panel_edges
            )
            comparison["nested_panel_reuse"] = bool(nested)
            comparison["added_radial_panel"] = [
                float(left_spec.u_max),
                float(right_spec.u_max),
            ]
            left_result = config_results[left_id]["pairings"].get(str(pairing), {})
            right_result = config_results[right_id]["pairings"].get(str(pairing), {})
            if (
                left_result.get("status") == "integrated"
                and right_result.get("status") == "integrated"
            ):
                comparison["signed_tail_increment_J_m2"] = float(
                    right_result["partial_free_energy_J_m2"]
                    - left_result["partial_free_energy_J_m2"]
                )
        record["cutoff_comparison_semantics"] = (
            "each transition adds exactly one new radial panel while retaining all "
            "earlier nodes and weights"
        )
    return output


__all__ = [
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "absolute_then_relative",
    "aggregate_certified_outer_q",
    "build_staged_grid_plan",
    "build_union_node_manifest",
    "compare_ladders",
]
