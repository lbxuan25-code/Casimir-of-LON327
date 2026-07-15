"""Diagnostic fixed/adaptive comparison using successive-high convergence semantics."""
from __future__ import annotations

from typing import Any

import numpy as np

import validation.commands.matsubara.arbitrary_q_vector_adaptive_compare as legacy
from lno327.workflows.arbitrary_q_vector_adaptive_successive import (
    ArbitraryQVectorAdaptiveSuccessiveOptions,
    ArbitraryQVectorAdaptiveSuccessiveResponseCache,
)
from lno327.workflows.arbitrary_q_vector_adaptive_successive_cached import (
    integrate_arbitrary_q_vector_adaptive_successive_cached,
)
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)


def _physical(result: object, q: np.ndarray, args: Any) -> list[dict[str, Any]]:
    config = OrbitAcceptancePhysicsConfig(
        separation_nm=args.separation_nm,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
    )
    rows: list[dict[str, Any]] = []
    for n, frequency, component, rhs in zip(
        args.matsubara_indices,
        result.xi_eV_values,
        result.components,
        result.rhs,
        strict=True,
    ):
        state = evaluate_matsubara_pipeline(
            components=component,
            rhs=rhs,
            q_model=q,
            xi_eV=float(frequency),
            config=config,
        )
        rows.append(
            {
                "n": int(n),
                "passed": bool(state["physical_passed"]),
                "ward": bool(state["ward_passed"]),
                "strict_static": bool(state["strict_static_ward_passed"]),
                "sheet": bool(state["sheet_validation_passed"]),
                "reflection_constructed": bool(state["reflection_constructed"]),
                "logdet_passed": bool(state["logdet_passed"]),
                "reflection": np.asarray(state["reflection"], dtype=complex),
                "logdet": float(state["logdet"]),
                "primary": np.asarray(state["primary_response"], dtype=complex),
                "ward_effective_mixed_ratio_max": float(
                    state["ward_effective_mixed_ratio_max"]
                ),
                "schur_condition_number": float(state["schur_condition_number"]),
                "error": str(state["error"]),
            }
        )
    return rows


def _physical_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": int(row["n"]),
        "passed": bool(row["passed"]),
        "ward": bool(row["ward"]),
        "strict_static": bool(row["strict_static"]),
        "sheet": bool(row["sheet"]),
        "reflection_constructed": bool(row["reflection_constructed"]),
        "logdet_passed": bool(row["logdet_passed"]),
        "logdet": float(row["logdet"]),
        "primary_norm": float(np.linalg.norm(row["primary"])),
        "reflection_norm": float(np.linalg.norm(row["reflection"])),
        "ward_effective_mixed_ratio_max": float(
            row["ward_effective_mixed_ratio_max"]
        ),
        "schur_condition_number": float(row["schur_condition_number"]),
        "error": str(row["error"]),
    }


def main(argv=None) -> None:
    legacy.ArbitraryQVectorAdaptiveOptions = ArbitraryQVectorAdaptiveSuccessiveOptions
    legacy.ArbitraryQVectorAdaptiveResponseCache = (
        ArbitraryQVectorAdaptiveSuccessiveResponseCache
    )
    legacy.integrate_arbitrary_q_vector_adaptive_cached = (
        integrate_arbitrary_q_vector_adaptive_successive_cached
    )
    legacy._physical = _physical
    legacy._physical_summary = _physical_summary
    legacy.main(argv)


if __name__ == "__main__":
    main()
