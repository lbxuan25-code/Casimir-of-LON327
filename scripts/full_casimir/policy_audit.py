from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._diagnostic_io import config_difference_paths, mapping


def _pick(source: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: source.get(name) for name in names if name in source}


def policy_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract numerical policy separately from physical model identity."""

    outer = mapping(config.get("outer_tail_config"))
    joint = mapping(outer.get("joint_config"))
    radial = mapping(joint.get("radial_config"))
    point = mapping(radial.get("point_config"))
    return {
        "microscopic_acceptance": _pick(
            point,
            "N_candidates",
            "required_consecutive_passes",
            "logdet_rtol",
            "logdet_atol",
            "shifts",
        ),
        "microscopic_sampling": _pick(
            point,
            "u_max_values",
            "radial_orders",
            "angular_orders",
            "angular_offsets",
        ),
        "radial_controller": _pick(
            radial,
            "initial_panel_edges",
            "radial_order",
            "angular_order",
            "radial_rtol",
            "radial_atol_J_m2",
            "max_refinement_rounds",
            "max_panel_depth",
            "refine_panels_per_round",
            "max_microscopic_q_nodes",
        ),
        "joint_controller": _pick(
            joint,
            "angular_orders",
            "primary_offset_fraction",
            "audit_offset_fraction",
            "outer_rtol",
            "outer_atol_J_m2",
            "radial_budget_fraction",
            "angular_budget_fraction",
            "offset_rtol",
            "offset_atol_J_m2",
            "initial_radial_round_cap",
            "radial_round_step",
            "max_joint_iterations",
            "max_total_microscopic_q_nodes",
        ),
        "outer_tail": _pick(
            outer,
            "cutoff_u_values",
            "total_outer_rtol",
            "total_outer_atol_J_m2",
            "finite_domain_budget_fraction",
            "tail_budget_fraction",
            "joint_budget_fraction_within_finite",
            "offset_budget_fraction_within_finite",
            "tail_start_u",
            "tail_window_shells",
            "tail_ratio_max",
            "shell_width_rtol",
            "shell_width_atol",
            "max_total_microscopic_q_nodes",
        ),
        "matsubara_tail": _pick(
            config,
            "matsubara_cutoff_values",
            "total_free_energy_rtol",
            "total_free_energy_atol_J_m2",
            "finite_matsubara_budget_fraction",
            "matsubara_tail_budget_fraction",
            "tail_start_n",
            "tail_window_terms",
            "tail_ratio_max",
            "max_total_microscopic_point_entries",
        ),
        "runtime_scheduling": {
            **_pick(
                point,
                "workers",
                "parallel_mode",
                "memory_budget_gb",
                "max_context_workers",
            ),
            **_pick(config, "certifier_q_batch_size"),
        },
        "excluded_physical_fields": {
            "point_config": [
                name
                for name in (
                    "pairings",
                    "temperature_K",
                    "separation_nm",
                    "plate_angles_deg",
                    "delta0_eV",
                    "eta_eV",
                    "degeneracy",
                    "matsubara_indices",
                )
                if name in point
            ],
            "reason": (
                "Physical model identity is excluded from numerical-policy parity. "
                "The audit compares gates, budgets, ladders, and controller rules."
            ),
        },
    }


def _value_at_path(payload: Any, path: str) -> Any:
    if not path.startswith("$"):
        return None
    tokens = path[1:].split(".")
    current = payload
    for token in tokens:
        if not token:
            continue
        if token == "length":
            return len(current) if isinstance(current, (list, tuple, Mapping)) else None
        while "[" in token:
            name, rest = token.split("[", 1)
            if name:
                current = mapping(current).get(name)
            position_token, token = rest.split("]", 1)
            try:
                position = int(position_token)
                current = current[position] if isinstance(current, list) else None
            except (ValueError, IndexError):
                return None
            token = token.lstrip(".")
        if token:
            current = mapping(current).get(token)
    return current


def _difference_category(path: str) -> str:
    if path.startswith("$.runtime_scheduling"):
        return "runtime_scheduling"
    if path.startswith("$.microscopic_acceptance"):
        return "acceptance_policy"
    if path.startswith("$.outer_tail") or path.startswith("$.matsubara_tail"):
        return "tail_and_error_budget_policy"
    if "budget_fraction" in path or path.startswith("$.joint_controller"):
        return "controller_and_budget_policy"
    if path.startswith("$.microscopic_sampling") or path.startswith("$.radial_controller"):
        return "adaptive_ladder_or_sampling_policy"
    return "other_numerical_policy"


def compare_policy_snapshots(
    named_configs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(named_configs) < 2:
        return {
            "schema": "pairing-blind-policy-audit-v1",
            "status": "insufficient_runs",
            "run_count": len(named_configs),
            "pairing_blind_scientific_policy": None,
            "comparisons": [],
        }
    snapshots = [(name, policy_snapshot(config)) for name, config in named_configs]
    reference_name, reference = snapshots[0]
    comparisons: list[dict[str, Any]] = []
    scientific_difference_count = 0
    scheduling_difference_count = 0
    for name, snapshot in snapshots[1:]:
        records = []
        for path in config_difference_paths(reference, snapshot):
            if path.startswith("$.excluded_physical_fields"):
                continue
            category = _difference_category(path)
            if category == "runtime_scheduling":
                scheduling_difference_count += 1
            else:
                scientific_difference_count += 1
            record = {
                "path": path,
                "category": category,
                "reference_value": _value_at_path(reference, path),
                "compared_value": _value_at_path(snapshot, path),
            }
            if path.endswith(".length"):
                parent_path = path[: -len(".length")]
                record["reference_sequence"] = _value_at_path(reference, parent_path)
                record["compared_sequence"] = _value_at_path(snapshot, parent_path)
            records.append(record)
        comparisons.append(
            {
                "reference_run": reference_name,
                "compared_run": name,
                "difference_count": len(records),
                "differences": records,
            }
        )
    return {
        "schema": "pairing-blind-policy-audit-v1",
        "status": "analyzed",
        "run_count": len(snapshots),
        "reference_run": reference_name,
        "pairing_blind_scientific_policy": scientific_difference_count == 0,
        "scientific_policy_difference_count": scientific_difference_count,
        "runtime_scheduling_difference_count": scheduling_difference_count,
        "comparisons": comparisons,
        "snapshots": {name: snapshot for name, snapshot in snapshots},
        "interpretation": (
            "Scientific parity requires identical acceptance gates, controller rules, "
            "error allocations, and adaptive ladders after physical fields are removed."
        ),
    }


__all__ = ["compare_policy_snapshots", "policy_snapshot"]
