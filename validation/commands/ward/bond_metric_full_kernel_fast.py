"""Optimized entry point for the d-wave full-kernel bond-metric audit.

The established command orchestration and output schema remain in
``bond_metric_full_kernel``.  This entry point substitutes only the drop-in
primitive evaluator, which batches chunk eigensystems and reuses midpoint thermal
density matrices.  The quadrature, primitive vector, Schur assembly, Ward audit,
and fail-closed metadata are unchanged.
"""

from __future__ import annotations

from validation.commands.ward import bond_metric_full_kernel as _reference
from validation.lib.dwave_iterated_adaptive_fast import (
    build_dwave_static_integrand_context,
)


def main() -> None:
    previous = _reference.build_dwave_static_integrand_context
    _reference.build_dwave_static_integrand_context = (
        build_dwave_static_integrand_context
    )
    try:
        _reference.main()
    finally:
        _reference.build_dwave_static_integrand_context = previous


if __name__ == "__main__":
    main()
