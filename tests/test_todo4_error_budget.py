from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

import numpy as np

from lno327.constants import C0, E2_OVER_HBAR, EV_TO_J, HBAR, SIGMA0
from lno327.casimir.adaptive_outer_tail import AdaptiveOuterTailCasimirResult
from lno327.casimir.certified_matsubara import (
    MATSUBARA_TAIL_CERTIFICATE_CONTRACT,
    run_certified_matsubara_casimir,
    validate_dyadic_matsubara_policy,
)
from lno327.casimir.certified_tail import (
    active_power_metric_contraction_certificate,
)
from lno327.casimir.production import build_full_casimir_config
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _Provider:
    def __init__(self) -> None:
        self.configs: list[tuple[int, ...]] = []
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0
        self._entries = {}

    def reconfigure(self, config) -> None:
        self.configs.append(tuple(config.matsubara_indices))
        self.cached_point_count = len(config.matsubara_indices)


class _OuterRunner:
    def __init__(self, term_function, error_function=lambda _n: 1e-16) -> None:
        self.term_function = term_function
        self.error_function = error_function

    def __call__(self, config, *, provider=None):
        point = config.joint_config.radial_config.point_config
        indices = tuple(point.matsubara_indices)
        values = np.asarray([self.term_function(n) for n in indices], dtype=float)
        errors = np.asarray([self.error_function(n) for n in indices], dtype=float)
        payload = {
            pairing: {
                "status": "integrated_with_outer_tail_bound",
                "partial_free_energy_J_m2": float(np.sum(values)),
                "contributions_J_m2": values.tolist(),
                "estimated_total_outer_errors_J_m2": errors.tolist(),
                "matsubara_indices": list(indices),
                "outer_tail_certificate_path": "analytic_passive_vacuum",
            }
            for pairing in point.pairings
        }
        return AdaptiveOuterTailCasimirResult(
            status="adaptive_finite_partial",
            config=config,
            cutoff_converged=True,
            outer_tail_estimated_flag=True,
            all_finite_domain_runs_converged=True,
            all_microscopic_nodes_certified=True,
            selected_u_max=config.cutoff_u_values[-1],
            pairing_results=MappingProxyType(payload),
            cutoff_records=(),
            shell_records=(),
            termination_reason="analytic_passive_vacuum_tail_bound_met",
            provider_statistics=MappingProxyType({"cached_point_count": len(indices)}),
        )


def _config():
    return build_full_casimir_config(
        pairings=("spm",),
        N_candidates=(128, 192, 256),
        matsubara_cutoff_values=(1, 3, 7, 15, 31, 63),
        matsubara_tail_start_n=4,
        matsubara_tail_window_terms=4,
        matsubara_tail_ratio_max=0.8,
        total_free_energy_rtol=0.0,
        total_free_energy_atol_J_m2=1e-6,
    )


def test_formal_matsubara_policy_rejects_legacy_nondyadic_ladder() -> None:
    try:
        validate_dyadic_matsubara_policy(
            (1, 3, 7, 11, 15, 23, 31),
            tail_start_n=8,
            tail_window_blocks=4,
        )
    except ValueError as exc:
        assert "dyadic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("legacy nondyadic ladder must be rejected")


def test_inverse_square_tail_passes_as_dyadic_blocks() -> None:
    result = run_certified_matsubara_casimir(
        _config(),
        provider=_Provider(),
        outer_tail_runner=_OuterRunner(lambda n: 1e-9 / (n + 1) ** 2),
    )
    assert result.matsubara_converged
    assert result.formal_policy_passed
    assert result.production_casimir_allowed
    assert result.selected_matsubara_cutoff == 63
    channel = result.pairing_results["spm"]
    assert channel["matsubara_tail_certificate_contract"] == (
        MATSUBARA_TAIL_CERTIFICATE_CONTRACT
    )
    assert channel["matsubara_tail_holdout_passed"] is True


def test_inverse_cube_tail_bound_exceeds_actual_remainder() -> None:
    scale = 1e-9
    result = run_certified_matsubara_casimir(
        _config(),
        provider=_Provider(),
        outer_tail_runner=_OuterRunner(lambda n: scale / (n + 1) ** 3),
    )
    assert result.matsubara_converged
    channel = result.pairing_results["spm"]
    estimated = float(channel["estimated_matsubara_tail_bound_J_m2"])
    # Integral comparison: sum_{k=65}^inf 1/k^3 <= integral_{64.5}^inf dx/x^3.
    rigorous_actual_upper = scale / (2.0 * 64.5**2)
    assert estimated >= rigorous_actual_upper


def test_holdout_block_spike_is_fail_closed() -> None:
    def term(n: int) -> float:
        return 1e-9 / (n + 1) ** 2 if n < 32 else 1e-9

    result = run_certified_matsubara_casimir(
        _config(),
        provider=_Provider(),
        outer_tail_runner=_OuterRunner(term),
    )
    assert not result.matsubara_converged
    assert not result.production_casimir_allowed
    assert result.termination_reason == "matsubara_block_decay_or_holdout_not_established"


def test_static_frobenius_false_negative_is_replaced_by_exact_spectral_norm() -> None:
    config = _config()
    point_config = config.outer_tail_config.joint_config.radial_config.point_config
    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    beta = point_config.static_energy_scale_eV * EV_TO_J * lattice / (HBAR * C0)
    gamma = E2_OVER_HBAR / SIGMA0
    q_norm = 1e-3
    target = 0.8
    lam = 2.0 * target / (1.0 - target)
    chi_bar = lam * q_norm * beta / (point_config.degeneracy * gamma)
    dbar_t = lam * q_norm / (point_config.degeneracy * gamma * beta)
    plate = {
        "sheet_validation_passed": True,
        "reflection_constructed": True,
        "reflection_norm": math.sqrt(2.0) * target,
        "q_crystal": [q_norm, 0.0],
        "chi_bar": chi_bar,
        "dbar_t": dbar_t,
    }
    point = {
        "pairing": "spm",
        "n": 0,
        "q_label": "q0",
        "sweet_spot": {"status": "established", "audit_N": 256},
        "history": [
            {
                "N": 256,
                "shifts": {
                    "shift_0": {
                        "hard_physical_passed": True,
                        "plate_1": dict(plate),
                        "plate_2": dict(plate),
                    }
                },
            }
        ],
    }
    provider = _Provider()
    provider._entries = {"spm|0|x|y": point}
    certificate = active_power_metric_contraction_certificate(
        provider,
        point_config=point_config,
    )
    assert plate["reflection_norm"] > 1.0
    assert certificate["all_points_certified"] is True
    assert abs(certificate["maximum_recorded_upper_bound"] - target) < 1e-12
    assert certificate["methods"] == ["exact_static_diagonal_spectral_norm"]


def test_error_budget_is_pairing_blind_and_explicit_in_config() -> None:
    config = _config()
    assert config.finite_matsubara_budget_fraction == 0.7
    assert config.matsubara_tail_budget_fraction == 0.3
    outer = config.outer_tail_config
    assert outer.finite_domain_budget_fraction == 0.7
    assert outer.tail_budget_fraction == 0.3
    assert outer.joint_budget_fraction_within_finite == 0.8
    assert outer.offset_budget_fraction_within_finite == 0.2


def test_production_plan_serializes_complete_error_budget() -> None:
    from scripts.full_casimir import scan

    args = scan._parser().parse_args(
        [
            "plan",
            "--pairings",
            "spm",
            "dwave",
            "--distances-nm",
            "20",
            "--angles-deg",
            "0",
        ]
    )
    plan = scan.build_scan_plan(
        args,
        code_identity={
            "git_commit": "a" * 40,
            "tracked_worktree_clean": True,
        },
    )
    policy = plan["scientific_policy"]
    budget = policy["error_budget"]
    assert budget["contract_version"] == "full-casimir-error-budget-v1"
    assert budget["pairing_blind"] is True
    assert budget["static_contraction_norm"] == "exact_spectral_norm"
    assert policy["matsubara"]["tail_estimator"] == (
        "dyadic_absolute_block_envelope"
    )
    assert policy["matsubara"]["per_term_ratio_acceptance_forbidden"] is True
    assert policy["production_authorization"][
        "requires_total_error_budget_closed"
    ] is True
