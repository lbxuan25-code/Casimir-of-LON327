"""Main-pipeline Lifshitz trace-log single-point helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .readiness import round_trip_factor, trace_log_integrand, trace_log_matrix


def trace_log_point(
    R1_te_tm: np.ndarray,
    R2_te_tm: np.ndarray,
    kappa_m_inv: float,
    separation_m: float,
) -> dict[str, Any]:
    """Return a single trace-log point for external Matsubara/Q summation."""

    return {
        "round_trip_factor": round_trip_factor(kappa_m_inv, separation_m),
        "trace_log_matrix": trace_log_matrix(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
        "logdet_integrand": trace_log_integrand(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
        "single_point_trace_log_for_main_pipeline": True,
        "does_not_by_itself_perform_full_integral": True,
        "used_by_main_pipeline_after_external_grid_sum": True,
    }


def lifshitz_integrand_metadata() -> dict[str, Any]:
    """Return trace-log metadata for the main Casimir pipeline."""

    return {
        "single_point_trace_log_for_main_pipeline": True,
        "does_not_by_itself_perform_full_integral": True,
        "used_by_main_pipeline_after_external_grid_sum": True,
        "formula": "log det[I - exp(-2*kappa*d) R1 @ R2]",
        "round_trip_factor_formula": "exp(-2*kappa*d)",
    }
