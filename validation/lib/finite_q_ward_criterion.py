"""Formal finite-q BdG Ward closure criterion.

The formal criterion is defined only for the full effective-action Hessian.  The
AA kernel is K_AA = bubble + direct, and the requested collective-Schur response
is checked by a homogeneous Ward contraction.  Bubble-only contact-RHS formulae
are intentionally not exposed here, so they cannot be mistaken for closure
criteria by downstream scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _row_field(row: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _criterion_q_key(q_model: Any) -> tuple[float, float]:
    q = tuple(float(value) for value in q_model)
    if len(q) != 2:
        raise ValueError("q_model must have two components")
    return q[0], q[1]


def _criterion_residual_vectors(row: Any) -> tuple[np.ndarray, np.ndarray] | None:
    left = _row_field(row, "left_ward_residual_vector")
    right = _row_field(row, "right_ward_residual_vector")
    if left is None or right is None:
        return None

    def to_array(payload: Any) -> np.ndarray:
        values: list[complex] = []
        for item in payload:
            if isinstance(item, dict):
                values.append(complex(float(item.get("real", 0.0)), float(item.get("imag", 0.0))))
            else:
                values.append(complex(getattr(item, "real", 0.0), getattr(item, "imag", 0.0)))
        return np.asarray(values, dtype=complex)

    return to_array(left), to_array(right)


def _empty_criterion_pairing(
    *,
    q0_status: str,
    closure_response_name: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "evaluated": False,
        "q0_precondition_status": q0_status,
        "closure_response_name": closure_response_name,
        "max_closure_primary_residual_norm": None,
        "max_closure_homogeneous_residual_norm": None,
        "closed": False,
        "blocking_response_name": closure_response_name,
        "blocking_q_model": None,
        "blocking_reason": reason,
        "rows": [],
        "valid_for_casimir_input": False,
    }


def _closure_row_payload(
    *,
    row: Any,
    absolute_tol: float,
    relative_tol: float,
) -> dict[str, Any]:
    homogeneous_norm = float(_row_field(row, "max_ward_residual_norm", float("nan")))
    threshold = float(absolute_tol + relative_tol * abs(homogeneous_norm))
    return {
        "q_model": [float(value) for value in _row_field(row, "q_model")],
        "response_name": str(_row_field(row, "response_name")),
        "criterion_type": "full_hessian_collective_schur",
        "homogeneous_residual_norm": homogeneous_norm,
        "primary_residual_norm": homogeneous_norm,
        "primary_residual_kind": "homogeneous_full_hessian_schur",
        "absolute_tol": float(absolute_tol),
        "relative_tol": float(relative_tol),
        "threshold": threshold,
        "passed": bool(homogeneous_norm <= threshold),
        "valid_for_casimir_input": False,
    }


def evaluate_finite_q_bdg_ward_criterion(
    *,
    finite_q_rows: list[dict[str, Any]] | tuple[Any, ...],
    pairings: tuple[str, ...],
    q_values: tuple[float, ...],
    closure_response_name: str = "amplitude_phase_schur",
    absolute_tol: float = 1e-6,
    relative_tol: float = 1e-6,
    q0_precondition_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate homogeneous full-Hessian finite-q BdG Ward closure."""

    if closure_response_name not in {"minus_schur", "amplitude_phase_schur"}:
        raise ValueError("closure_response_name must be minus_schur or amplitude_phase_schur")

    rows_by_key: dict[tuple[str, tuple[float, float], str], Any] = {}
    for row in finite_q_rows:
        try:
            key = (
                str(_row_field(row, "pairing_name")),
                _criterion_q_key(_row_field(row, "q_model")),
                str(_row_field(row, "response_name")),
            )
        except (TypeError, ValueError):
            continue
        rows_by_key[key] = row

    by_pairing: dict[str, Any] = {}
    largest_blocker: dict[str, Any] | None = None
    closed_pairings: list[str] = []
    failed_pairings: list[str] = []

    for pairing in pairings:
        q0_status = (q0_precondition_status or {}).get(pairing, "unknown")
        if q0_status in {"diagnostic_only_not_passed", "missing_q0_status"}:
            payload = _empty_criterion_pairing(
                q0_status=q0_status,
                closure_response_name=closure_response_name,
                reason="q0_precondition_not_established",
            )
            by_pairing[pairing] = payload
            failed_pairings.append(pairing)
            continue

        criterion_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        missing_vectors: list[str] = []
        closure_failures: list[dict[str, Any]] = []
        max_closure_primary: float | None = None
        max_closure_homogeneous: float | None = None

        for q_value in q_values:
            q_key = _criterion_q_key((float(q_value), 0.0))
            row = rows_by_key.get((pairing, q_key, closure_response_name))
            if row is None:
                missing.append(f"{closure_response_name}@q={list(q_key)}")
                continue
            if _criterion_residual_vectors(row) is None:
                missing_vectors.append(f"{closure_response_name}@q={list(q_key)}")
                continue
            payload = _closure_row_payload(row=row, absolute_tol=absolute_tol, relative_tol=relative_tol)
            criterion_rows.append(payload)
            max_closure_primary = (
                payload["primary_residual_norm"]
                if max_closure_primary is None
                else max(max_closure_primary, payload["primary_residual_norm"])
            )
            max_closure_homogeneous = (
                payload["homogeneous_residual_norm"]
                if max_closure_homogeneous is None
                else max(max_closure_homogeneous, payload["homogeneous_residual_norm"])
            )
            if not payload["passed"]:
                closure_failures.append(payload)

        evaluated = not missing and not missing_vectors and bool(criterion_rows)
        closed = bool(evaluated and not closure_failures)
        if not evaluated:
            reason = "missing_response_row" if missing else "missing_residual_vector"
            blocking_response = closure_response_name
            blocking_q = None
        elif closure_failures:
            worst = max(closure_failures, key=lambda item: float(item["primary_residual_norm"]))
            reason = "closure_primary_residual_above_tolerance"
            blocking_response = str(worst["response_name"])
            blocking_q = list(worst["q_model"])
        else:
            reason = "closed"
            blocking_response = None
            blocking_q = None

        payload = {
            "evaluated": evaluated,
            "q0_precondition_status": q0_status,
            "closure_response_name": closure_response_name,
            "max_closure_primary_residual_norm": max_closure_primary,
            "max_closure_homogeneous_residual_norm": max_closure_homogeneous,
            "closed": closed,
            "blocking_response_name": blocking_response,
            "blocking_q_model": blocking_q,
            "blocking_reason": reason,
            "missing": missing,
            "missing_residual_vectors": missing_vectors,
            "rows": criterion_rows,
            "valid_for_casimir_input": False,
        }
        by_pairing[pairing] = payload
        if closed:
            closed_pairings.append(pairing)
        else:
            failed_pairings.append(pairing)
            blockers = [row for row in criterion_rows if row.get("primary_residual_norm") is not None]
            if blockers:
                worst = max(blockers, key=lambda item: float(item["primary_residual_norm"]))
                candidate = {
                    "pairing_name": pairing,
                    "q_model": list(worst["q_model"]),
                    "response_name": str(worst["response_name"]),
                    "primary_residual_norm": float(worst["primary_residual_norm"]),
                }
                if largest_blocker is None or candidate["primary_residual_norm"] > float(
                    largest_blocker["primary_residual_norm"]
                ):
                    largest_blocker = candidate

    evaluated = bool(by_pairing) and all(payload.get("evaluated", False) for payload in by_pairing.values())
    ward_identity_closed = bool(evaluated and not failed_pairings)
    if not evaluated:
        recommended = "Ensure finite_q_rows include the requested full-Hessian closure response for every pairing and q."
    elif failed_pairings:
        recommended = "Inspect the largest homogeneous full-Hessian Schur Ward residual."
    else:
        recommended = "No finite-q BdG Ward closure fix is indicated by the full-Hessian criterion."

    return {
        "criterion_version": "full_hessian_v1",
        "criterion_formal_name": "full_hessian_v1",
        "aa_kernel_definition": "K_AA_full = K_AA_bubble + K_AA_direct",
        "closure_identity": "W(K_AA_full - K_Aeta inv(K_etaeta) K_etaA) = 0",
        "direct_role": "included_in_full_hessian",
        "closure_response_name": closure_response_name,
        "absolute_tol": float(absolute_tol),
        "relative_tol": float(relative_tol),
        "evaluated": evaluated,
        "ward_identity_closed": ward_identity_closed,
        "valid_for_casimir_input": False,
        "by_pairing": by_pairing,
        "summary": {
            "closed_pairings": closed_pairings,
            "failed_pairings": failed_pairings,
            "largest_blocker": largest_blocker,
            "recommended_next_fix": recommended,
            "valid_for_casimir_input": False,
        },
    }
