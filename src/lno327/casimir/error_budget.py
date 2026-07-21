"""Single pairing-blind production error-budget contract."""
from __future__ import annotations

from typing import Any

ERROR_BUDGET_CONTRACT_VERSION = "full-casimir-error-budget-v1"
FINITE_MATSUBARA_BUDGET_FRACTION = 0.7
MATSUBARA_TAIL_BUDGET_FRACTION = 0.3
OUTER_FINITE_BUDGET_FRACTION = 0.7
OUTER_TAIL_BUDGET_FRACTION = 0.3
JOINT_BUDGET_FRACTION_WITHIN_OUTER_FINITE = 0.8
OFFSET_BUDGET_FRACTION_WITHIN_OUTER_FINITE = 0.2
MATSUBARA_HOLDOUT_BLOCKS = 1


def error_budget_policy_payload(*, radial_budget_fraction: float) -> dict[str, Any]:
    radial = float(radial_budget_fraction)
    if not 0.0 < radial < 1.0:
        raise ValueError("radial_budget_fraction must lie strictly between zero and one")
    return {
        "contract_version": ERROR_BUDGET_CONTRACT_VERSION,
        "pairing_blind": True,
        "finite_matsubara_fraction": FINITE_MATSUBARA_BUDGET_FRACTION,
        "matsubara_tail_fraction": MATSUBARA_TAIL_BUDGET_FRACTION,
        "outer_finite_fraction_per_term": OUTER_FINITE_BUDGET_FRACTION,
        "outer_tail_fraction_per_term": OUTER_TAIL_BUDGET_FRACTION,
        "joint_fraction_within_outer_finite": (
            JOINT_BUDGET_FRACTION_WITHIN_OUTER_FINITE
        ),
        "offset_fraction_within_outer_finite": (
            OFFSET_BUDGET_FRACTION_WITHIN_OUTER_FINITE
        ),
        "radial_fraction_within_joint": radial,
        "angular_fraction_within_joint": 1.0 - radial,
        "matsubara_tail_estimator": "dyadic_absolute_block_envelope",
        "matsubara_holdout_blocks": MATSUBARA_HOLDOUT_BLOCKS,
        "per_term_ratio_is_diagnostic_only": True,
        "outer_tail_paths": [
            "analytic_passive_vacuum",
            "geometric_numerical_shell_envelope",
        ],
        "static_contraction_norm": "exact_spectral_norm",
        "frobenius_norm_is_acceptance_gate": False,
        "acceptance_requires_total_budget_closure": True,
    }


__all__ = [
    "ERROR_BUDGET_CONTRACT_VERSION",
    "FINITE_MATSUBARA_BUDGET_FRACTION",
    "JOINT_BUDGET_FRACTION_WITHIN_OUTER_FINITE",
    "MATSUBARA_HOLDOUT_BLOCKS",
    "MATSUBARA_TAIL_BUDGET_FRACTION",
    "OFFSET_BUDGET_FRACTION_WITHIN_OUTER_FINITE",
    "OUTER_FINITE_BUDGET_FRACTION",
    "OUTER_TAIL_BUDGET_FRACTION",
    "error_budget_policy_payload",
]
