"""Formal finite-q BdG Ward closure criterion.

The formal criterion is defined for the full effective-action Hessian.  In
particular, the AA kernel is the full Hessian K_AA = bubble + direct, and the
final collective-Schur response is checked by a homogeneous Ward contraction.
Bubble-only contact-RHS residuals may be reported as diagnostics, but they are
not used as the primary closure criterion.
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


def _criterion_vector_payload(vector: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
            "abs": float(abs(value)),
        }
        for value in np.asarray(vector, dtype=complex).reshape(-1)
    ]


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


def _criterion_norm(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(np.linalg.norm(np.asarray(left, dtype=complex)), np.linalg.norm(np.asarray(right, dtype=complex))))


def _criterion_row_payload(
    *,
    row: Any,
    criterion_type: str,
    homogeneous_residual_norm: float,
    primary_residual_norm: float,
    primary_residual_kind: str,
    direct_residual_norm: float | None,
    bubble_only_left: np.ndarray | None = None,
    bubble_only_right: np.ndarray | None = None,
    plus_direct_left: np.ndarray | None = None,
    plus_direct_right: np.ndarray | None = None,
    absolute_tol: float,
    relative_tol: float,
) -> dict[str, Any]:
    scale = max(abs(homogeneous_residual_norm), abs(direct_residual_norm or 0.0))
    threshold = float(absolute_tol + relative_tol * scale)
    payload = {
        "q_model": [float(value) for value in _row_field(row, "q_model")],
        "response_name": str(_row_field(row, "response_name")),
        "criterion_type": criterion_type,
        "homogeneous_residual_norm": float(homogeneous_residual_norm),
        "primary_residual_norm": float(primary_residual_norm),
        "primary_residual_kind": primary_residual_kind,
        "direct_residual_norm": None if direct_residual_norm is None else float(direct_residual_norm),
        "absolute_tol": float(absolute_tol),
        "relative_tol": float(relative_tol),
        "threshold": threshold,
        "passed": bool(primary_residual_norm <= threshold),
        "valid_for_casimir_input": False,
    }
    if bubble_only_left is not None and bubble_only_right is not None:
        bubble_only_norm = _criterion_norm(bubble_only_left, bubble_only_right)
        payload["bubble_only_residual_norm"] = bubble_only_norm
        payload["minus_direct_residual_norm"] = bubble_only_norm
        payload["bubble_only_left_residual_vector"] = _criterion_vector_payload(bubble_only_left)
        payload["bubble_only_right_residual_vector"] = _criterion_vector_payload(bubble_only_right)
        payload["minus_direct_left_residual_vector"] = payload["bubble_only_left_residual_vector"]
        payload["minus_direct_right_residual_vector"] = payload["bubble_only_right_residual_vector"]
        payload["diagnostic_only_minus_direct"] = True
    if plus_direct_left is not None and plus_direct_right is not None:
        payload["plus_direct_residual_norm"] = _criterion_norm(plus_direct_left, plus_direct_right)
        payload["plus_direct_left_residual_vector"] = _criterion_vector_payload(plus_direct_left)
        payload["plus_direct_right_residual_vector"] = _criterion_vector_payload(plus_direct_right)
        payload["diagnostic_only_plus_direct"] = True
    return payload


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
        "max_closure_bubble_only_residual_norm": None,
        "max_direct_residual_norm": None,
        "closed": False,
        "blocking_response_name": closure_response_name,
        "blocking_q_model": None,
        "blocking_reason": reason,
        "rows": [],
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
    """Evaluate full-Hessian finite-q BdG Ward closure from compact scan rows."""

    required_responses = ("bare_bubble", "direct", "bare_total", "minus_schur", "amplitude_phase_schur")
    criterion_types = {
        "bare_bubble": "bubble_only_diagnostic",
        "bare_total": "full_hessian_aa",
        "minus_schur": "collective_corrected_intermediate",
        "amplitude_phase_schur": "collective_corrected_final",
    }
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
        max_closure_bubble_only: float | None = None
        max_direct_residual: float | None = None

        for q_value in q_values:
            q_key = _criterion_q_key((float(q_value), 0.0))
            found = {name: rows_by_key.get((pairing, q_key, name)) for name in required_responses}
            for name, row in found.items():
                if row is None:
                    missing.append(f"{name}@q={list(q_key)}")
            if any(row is None for row in found.values()):
                continue

            direct_vectors = _criterion_residual_vectors(found["direct"])
            if direct_vectors is None:
                missing_vectors.append(f"direct@q={list(q_key)}")
                continue
            direct_left, direct_right = direct_vectors
            direct_norm = _criterion_norm(direct_left, direct_right)
            max_direct_residual = direct_norm if max_direct_residual is None else max(max_direct_residual, direct_norm)

            for response_name in ("bare_bubble", "bare_total", "minus_schur", "amplitude_phase_schur"):
                row = found[response_name]
                vectors = _criterion_residual_vectors(row)
                homogeneous_norm = float(_row_field(row, "max_ward_residual_norm", float("nan")))
                if vectors is None:
                    missing_vectors.append(f"{response_name}@q={list(q_key)}")
                    continue
                left, right = vectors

                bubble_only_left = left - direct_left
                bubble_only_right = right - direct_right
                plus_direct_left = left + direct_left
                plus_direct_right = right + direct_right

                primary_norm = homogeneous_norm
                primary_kind = "homogeneous_full_hessian" if response_name == "bare_total" else "homogeneous"
                if response_name == "bare_bubble":
                    primary_kind = "homogeneous_bubble_only_diagnostic"
                    bubble_only_left = None
                    bubble_only_right = None
                    plus_direct_left = None
                    plus_direct_right = None

                payload = _criterion_row_payload(
                    row=row,
                    criterion_type=criterion_types[response_name],
                    homogeneous_residual_norm=homogeneous_norm,
                    primary_residual_norm=primary_norm,
                    primary_residual_kind=primary_kind,
                    direct_residual_norm=direct_norm if response_name != "bare_bubble" else None,
                    bubble_only_left=bubble_only_left,
                    bubble_only_right=bubble_only_right,
                    plus_direct_left=plus_direct_left if response_name != "bare_bubble" else None,
                    plus_direct_right=plus_direct_right if response_name != "bare_bubble" else None,
                    absolute_tol=absolute_tol,
                    relative_tol=relative_tol,
                )
                criterion_rows.append(payload)

                if response_name == closure_response_name:
                    max_closure_primary = (
                        payload["primary_residual_norm"]
                        if max_closure_primary is None
                        else max(max_closure_primary, payload["primary_residual_norm"])
                    )
                    max_closure_homogeneous = (
                        homogeneous_norm if max_closure_homogeneous is None else max(max_closure_homogeneous, homogeneous_norm)
                    )
                    if payload.get("bubble_only_residual_norm") is not None:
                        max_closure_bubble_only = (
                            payload["bubble_only_residual_norm"]
                            if max_closure_bubble_only is None
                            else max(max_closure_bubble_only, payload["bubble_only_residual_norm"])
                        )
                    if not payload["passed"]:
                        closure_failures.append(payload)

        evaluated = not missing and not missing_vectors and bool(criterion_rows)
        closed = bool(evaluated and not closure_failures)
        if not evaluated:
            reason = "missing_response_row" if missing else "missing_residual_vector"
            blocking_response = "direct" if any(item.startswith("direct@") for item in missing_vectors) else closure_response_name
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
            "max_closure_bubble_only_residual_norm": max_closure_bubble_only,
            "max_closure_contact_aware_residual_norm": max_closure_bubble_only,
            "max_direct_residual_norm": max_direct_residual,
            "max_contact_rhs_norm": max_direct_residual,
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
            blockers = [
                row
                for row in criterion_rows
                if row["response_name"] == closure_response_name and row.get("primary_residual_norm") is not None
            ]
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
        recommended = "Ensure finite_q_rows include full-Hessian closure residual vectors for every requested pairing and q."
    elif failed_pairings:
        recommended = "Inspect the largest homogeneous full-Hessian finite-q BdG Ward residual."
    else:
        recommended = "No finite-q BdG Ward closure fix is indicated by the full-Hessian criterion."

    return {
        "criterion_version": "contact_aware_v1",
        "criterion_formal_name": "full_hessian_v1",
        "aa_kernel_definition": "K_AA_full = K_AA_bubble + K_AA_direct",
        "closure_identity": "W(K_AA_full - K_Aeta inv(K_etaeta) K_etaA) = 0",
        "direct_role": "included_in_full_hessian_not_subtracted_from_formal_closure",
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
