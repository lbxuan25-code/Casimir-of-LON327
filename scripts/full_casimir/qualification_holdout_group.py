from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.fixed_outer_q import OuterQNodeManifest
from lno327.casimir.strict_transverse_runner import run_strict_transverse_certifier

from .cache_migration import _point_config_from_run_config, _read_json_mapping
from .config import case_name


def group_id(key: tuple[Any, ...], items: Sequence[Mapping[str, Any]]) -> str:
    from .data_management import _digest

    return _digest({"key": list(key), "identities": [list(x["identity"]) for x in items]})


def build_groups(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int, int], list[Mapping[str, Any]]] = {}
    for item in plan.get("items", []):
        if not isinstance(item, Mapping):
            raise ValueError("holdout plan contains a malformed item")
        holdout = item.get("holdout_N")
        if not isinstance(holdout, list) or len(holdout) != 2:
            raise ValueError("holdout item must contain two N levels")
        key = (
            str(item["pairing"]),
            int(item["n"]),
            int(item["candidate_audit_N"]),
            int(holdout[0]),
            int(holdout[1]),
        )
        grouped.setdefault(key, []).append(item)
    output = []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda x: tuple(str(v) for v in x["identity"]))
        output.append({"group_id": group_id(key, ordered), "key": list(key), "items": ordered})
    if not output:
        raise ValueError("holdout plan contains no execution groups")
    return output


def _evaluate(
    items: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    point_rows: Sequence[Mapping[str, Any]],
    holdout_levels: tuple[int, int],
) -> list[dict[str, Any]]:
    by_label = {str(row.get("q_label")): row for row in point_rows if isinstance(row, Mapping)}
    results = []
    for label, item in zip(labels, items, strict=True):
        row = by_label.get(label)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"holdout certifier omitted {label}")
        history = {
            int(x["N"]): x for x in row.get("history", []) if isinstance(x, Mapping)
        }
        candidate = item["candidate_values_by_shift"]
        threshold = float(item["acceptance_threshold"])
        levels = []
        point_passed = True
        for N in holdout_levels:
            level = history.get(N)
            if not isinstance(level, Mapping):
                raise RuntimeError(f"holdout history omitted N={N} for {label}")
            shifts = level.get("shifts")
            if not isinstance(shifts, Mapping) or set(shifts) != set(candidate):
                raise RuntimeError(f"holdout shifts differ from candidate for {label}")
            hard = True
            deltas = []
            for shift, state in shifts.items():
                if not isinstance(state, Mapping):
                    hard = False
                    continue
                hard = hard and bool(state.get("hard_physical_passed"))
                deltas.append(abs(float(state["two_plate_logdet"]) - float(candidate[shift])))
            delta = max(deltas, default=float("inf"))
            passed = bool(hard and np.isfinite(delta) and delta <= threshold)
            point_passed = point_passed and passed
            levels.append(
                {
                    "N": N,
                    "all_hard_physical_gates_passed": hard,
                    "maximum_shiftwise_absolute_delta": delta,
                    "acceptance_threshold": threshold,
                    "passed": passed,
                }
            )
        results.append(
            {
                "identity": list(item["identity"]),
                "reasons": list(item["reasons"]),
                "predicted_local_uncertainty": item["predicted_local_uncertainty"],
                "safety_factor": item["safety_factor"],
                "levels": levels,
                "passed": point_passed,
            }
        )
    return results


def run_group(
    group: Mapping[str, Any],
    *,
    output_root: Path,
    profile: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    pairing, n, candidate_N, holdout_1, holdout_2 = group["key"]
    items = list(group["items"])
    target = Path(output_root) / case_name(str(pairing), 0, profile=profile)
    full = _read_json_mapping(target / "config.json", label="target config")
    point = _point_config_from_run_config(full)
    config = replace(
        point,
        pairings=(str(pairing),),
        matsubara_indices=(int(n),),
        N_candidates=(int(candidate_N), int(holdout_1), int(holdout_2)),
        required_consecutive_passes=2,
        logdet_rtol=0.0,
        logdet_atol=0.0,
        workers=int(policy["workers_per_group"]),
        parallel_mode=str(policy["parallel_mode_per_group"]),
        memory_budget_gb=float(policy["memory_budget_gb_per_group"]),
        max_context_workers=int(policy["max_context_workers_per_group"]),
        transverse_checkpoint_path=None,
    )
    labels = tuple(
        f"holdout_{str(group['group_id'])[:12]}_{i:04d}" for i in range(len(items))
    )
    manifest = OuterQNodeManifest(
        labels=labels,
        q_model=np.asarray([x["q_model"] for x in items], dtype=float),
        grids={},
        labels_by_spec={},
    )
    with TemporaryDirectory(prefix="lno327-holdout-group-") as temp:
        started = perf_counter()
        cert = run_strict_transverse_certifier(config, manifest, Path(temp) / "result.json")
        wall = perf_counter() - started
    results = _evaluate(
        items,
        labels,
        cert.payload.get("point_results", []),
        (int(holdout_1), int(holdout_2)),
    )
    return {
        "group_id": str(group["group_id"]),
        "group": [str(pairing), int(n), int(candidate_N), int(holdout_1), int(holdout_2)],
        "point_count": len(items),
        "wall_seconds": wall,
        "execution_levels": cert.payload.get("execution_levels", []),
        "stdout_tail": cert.stdout[-4000:],
        "stderr_tail": cert.stderr[-4000:],
        "results": results,
        "all_points_passed": bool(results) and all(bool(x["passed"]) for x in results),
    }
