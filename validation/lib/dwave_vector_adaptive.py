"""Validation helpers for exact-static vector-valued d-wave cubature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_vector_adaptive_cubature import (
    DWaveCubatureCell,
    cubature_cell_gauss_rule,
    merge_cell_components_before_schur,
    primitive_component_vector,
    primitive_ward_residual_vector,
    vector_error_metrics,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_shift_batch import (
    portable_shift_result,
    postprocess_merged,
    restore_portable_shift_result,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


@dataclass(frozen=True)
class VectorAdaptiveConfig:
    low_order: int
    high_order: int
    qx: float
    qy: float
    temperature_K: float
    delta0_eV: float
    eta_eV: float
    relative_tolerance: float
    absolute_tolerance: float
    ward_tolerance: float
    ward_absolute_tolerance: float
    condition_max: float
    raw_longitudinal_ceiling: float
    longitudinal_tolerance: float
    mixing_tolerance: float
    reality_tolerance: float
    passivity_tolerance: float
    separation_nm: float

    @property
    def q(self) -> np.ndarray:
        return np.asarray([self.qx, self.qy], dtype=float)


def _evaluate_rule(
    config: VectorAdaptiveConfig,
    cell: DWaveCubatureCell,
    order: int,
) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(config.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        output_si=False,
    )
    points, weights = cubature_cell_gauss_rule(cell, order)
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        kubo,
        pairing,
        FiniteQEngineOptions(),
    )
    workspace = precompute_finite_q_q_workspace(material, config.q)
    return {
        "components": finite_q_bdg_response_from_q_workspace(workspace, 0.0),
        "rhs": primitive_ward_rhs_from_q_workspace(workspace, 0.0),
        "workspace": workspace,
        "num_points": int(len(points)),
    }


def evaluate_cubature_cell(
    config: VectorAdaptiveConfig,
    cell: DWaveCubatureCell,
) -> dict[str, Any]:
    """Evaluate low/high rules for one cell and retain one template workspace."""

    low = _evaluate_rule(config, cell, int(config.low_order))
    high = _evaluate_rule(config, cell, int(config.high_order))
    return {
        "cell": cell,
        "low_components": low["components"],
        "low_rhs": low["rhs"],
        "high_components": high["components"],
        "high_rhs": high["rhs"],
        "workspace": high["workspace"],
        "evaluation_points": int(low["num_points"] + high["num_points"]),
    }


def portable_cubature_cell_result(result: dict[str, Any]) -> dict[str, Any]:
    """Strip immutable metadata before returning one cell through ProcessPool."""

    low = portable_shift_result(
        {
            "index": 0,
            "shift": np.asarray([0.0, 0.0]),
            "components": result["low_components"],
            "rhs": result["low_rhs"],
            "workspace": None,
        }
    )
    high = portable_shift_result(
        {
            "index": 0,
            "shift": np.asarray([0.0, 0.0]),
            "components": result["high_components"],
            "rhs": result["high_rhs"],
            "workspace": None,
        }
    )
    cell: DWaveCubatureCell = result["cell"]
    return {
        "cell": {
            "x0": float(cell.x0),
            "x1": float(cell.x1),
            "y0": float(cell.y0),
            "y1": float(cell.y1),
            "level": int(cell.level),
        },
        "low": low,
        "high": high,
        "evaluation_points": int(result["evaluation_points"]),
    }


def evaluate_cubature_cell_portable(
    config: VectorAdaptiveConfig,
    cell: DWaveCubatureCell,
) -> dict[str, Any]:
    return portable_cubature_cell_result(evaluate_cubature_cell(config, cell))


def restore_portable_cubature_cell_result(payload: dict[str, Any]) -> dict[str, Any]:
    low = restore_portable_shift_result(dict(payload["low"]))
    high = restore_portable_shift_result(dict(payload["high"]))
    return {
        "cell": DWaveCubatureCell(**dict(payload["cell"])),
        "low_components": low["components"],
        "low_rhs": low["rhs"],
        "high_components": high["components"],
        "high_rhs": high["rhs"],
        "workspace": None,
        "evaluation_points": int(payload["evaluation_points"]),
    }


def aggregate_cubature_cells(
    results: list[dict[str, Any]],
    template_workspace,
    config: VectorAdaptiveConfig,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Merge accepted high-rule primitives and estimate complete-vector errors."""

    high_components = [item["high_components"] for item in results]
    high_rhs = [item["high_rhs"] for item in results]
    merged_components, merged_rhs = merge_cell_components_before_schur(
        high_components,
        high_rhs,
        template_workspace,
        omega_eV=0.0,
    )
    physical = postprocess_merged(merged_components, merged_rhs, config)
    low_vectors = [
        primitive_component_vector(item["low_components"], item["low_rhs"])
        for item in results
    ]
    high_vectors = [
        primitive_component_vector(item["high_components"], item["high_rhs"])
        for item in results
    ]
    low_ward = [
        primitive_ward_residual_vector(item["low_components"], item["low_rhs"])
        for item in results
    ]
    high_ward = [
        primitive_ward_residual_vector(item["high_components"], item["high_rhs"])
        for item in results
    ]
    global_scale = max(
        float(np.linalg.norm(np.sum(np.stack(high_vectors), axis=0))),
        float(np.linalg.norm(merged_rhs.left)),
        float(np.linalg.norm(merged_rhs.right)),
        1e-30,
    )
    ward_threshold = float(config.ward_absolute_tolerance) + float(
        config.ward_tolerance
    ) * global_scale
    errors = vector_error_metrics(
        low_vectors,
        high_vectors,
        relative_tolerance=config.relative_tolerance,
        absolute_tolerance=config.absolute_tolerance,
        low_ward_vectors=low_ward,
        high_ward_vectors=high_ward,
        ward_threshold=ward_threshold,
    )
    return physical, np.asarray(errors.pop("cell_scores"), dtype=float), errors


def choose_refinement_indices(
    results: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    refine_fraction: float,
    min_refine_cells: int,
    max_level: int,
    max_cells: int,
    remaining_evaluation_points: int,
    points_per_child: int,
) -> list[int]:
    """Choose highest-error refinable cells without exceeding hard budgets."""

    if scores.shape != (len(results),):
        raise ValueError("scores must align with results")
    candidates = [
        index
        for index, item in enumerate(results)
        if int(item["cell"].level) < int(max_level)
    ]
    candidates.sort(key=lambda index: float(scores[index]), reverse=True)
    requested = max(int(min_refine_cells), int(np.ceil(float(refine_fraction) * len(results))))
    requested = min(requested, len(candidates))
    cell_capacity = max(0, (int(max_cells) - len(results)) // 3)
    point_capacity = max(0, int(remaining_evaluation_points) // (4 * int(points_per_child)))
    count = min(requested, cell_capacity, point_capacity)
    return candidates[:count]


__all__ = [
    "VectorAdaptiveConfig",
    "aggregate_cubature_cells",
    "choose_refinement_indices",
    "evaluate_cubature_cell",
    "evaluate_cubature_cell_portable",
    "portable_cubature_cell_result",
    "restore_portable_cubature_cell_result",
]
