"""Compatibility facade for the retained fixed-outer/adaptive-inner integrator.

The implementation now lives in :mod:`lno327.numerics.fixed_outer_adaptive_inner`
so validation and future microscopic workflows share one numerical definition.
"""

from lno327.numerics.fixed_outer_adaptive_inner import (
    EvaluationBudgetExceeded,
    FixedOuterAdaptiveInnerOptions,
    FixedOuterAdaptiveInnerOrientationResult,
    IntegrationOrder,
    integrate_fixed_outer_adaptive_inner_orientation,
)

GaussAdaptiveOptions = FixedOuterAdaptiveInnerOptions
GaussAdaptiveResult = FixedOuterAdaptiveInnerOrientationResult
gauss_outer_adaptive_integral = integrate_fixed_outer_adaptive_inner_orientation

__all__ = [
    "EvaluationBudgetExceeded",
    "GaussAdaptiveOptions",
    "GaussAdaptiveResult",
    "IntegrationOrder",
    "gauss_outer_adaptive_integral",
]
