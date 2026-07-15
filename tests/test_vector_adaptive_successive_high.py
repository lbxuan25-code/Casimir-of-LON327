from __future__ import annotations

import numpy as np

from lno327.workflows.arbitrary_q_vector_adaptive import ArbitraryQVectorAdaptiveOptions
from lno327.workflows.arbitrary_q_vector_adaptive_successive import (
    ArbitraryQVectorAdaptiveSuccessiveOptions,
    successive_high_controller_converged,
    successive_high_error_metrics,
    update_successive_high_streak,
)
from validation.__main__ import resolve_command


def test_successive_options_isolate_numerical_definition() -> None:
    legacy = ArbitraryQVectorAdaptiveOptions()
    successive = ArbitraryQVectorAdaptiveSuccessiveOptions()

    successive.validate()
    assert successive.fingerprint != legacy.fingerprint
    definition = successive.numerical_definition()
    assert definition["contract"] == "ArbitraryQVectorAdaptiveContract-v3-successive-high"
    assert definition["successive_stable_iterations"] == 2
    assert definition["same_iteration_low_high_is_diagnostic_only"] is True
    assert definition["local_ward_is_diagnostic_only"] is True


def test_successive_high_error_uses_complete_accepted_vectors() -> None:
    previous = np.zeros(43, dtype=complex)
    current = np.zeros(43, dtype=complex)
    previous[0] = 1.0
    current[0] = 1.0 + 5e-4

    metrics = successive_high_error_metrics(
        previous,
        current,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
    )

    assert metrics["global_group_error_ratio_max"] < 1.0


def test_successive_stability_requires_consecutive_passes() -> None:
    streak = update_successive_high_streak(0, 0.8)
    assert streak == 1
    streak = update_successive_high_streak(streak, 0.9)
    assert streak == 2
    streak = update_successive_high_streak(streak, 1.1)
    assert streak == 0
    assert update_successive_high_streak(2, float("nan")) == 0


def test_controller_requires_stability_and_integrated_ward() -> None:
    assert not successive_high_controller_converged(
        stable_streak=1,
        required_streak=2,
        high_rule_ward_passed=True,
    )
    assert not successive_high_controller_converged(
        stable_streak=2,
        required_streak=2,
        high_rule_ward_passed=False,
    )
    assert successive_high_controller_converged(
        stable_streak=2,
        required_streak=2,
        high_rule_ward_passed=True,
    )


def test_public_adaptive_compare_route_uses_successive_controller() -> None:
    assert resolve_command(
        "diagnostic", "arbitrary-q-vector-adaptive-compare"
    ) == (
        "validation.commands.matsubara."
        "arbitrary_q_vector_adaptive_successive_compare"
    )
