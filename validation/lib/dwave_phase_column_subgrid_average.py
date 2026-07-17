"""Average complementary phase-column quadratures before forming Ward ratios.

For an odd commensurate shift component, ``k +/- q/2`` lie on the complementary
half-step sublattice while the q=0 counterterm is sampled on the original grid.
Running the same physical q on complementary grid origins and averaging the raw
phase-column pieces restores a quadrature that treats both sublattices symmetrically.

The average must be taken before inferring the required counterterm multiplier:
the electromagnetic contraction, finite-q phase-bubble rotation, and q=0
counterterm rotation are averaged separately.  Averaging already-formed ratios is
not equivalent when the counterterm curvature differs slightly between subgrids.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
)


def _mean_complex(values: Sequence[complex]) -> complex:
    array = np.asarray([complex(value) for value in values], dtype=complex)
    if array.size == 0 or not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError("subgrid average requires finite non-empty complex values")
    return complex(np.mean(array))


def average_dwave_phase_column_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Average full or reduced commensurate phase-column payloads componentwise."""

    if len(payloads) < 2:
        raise ValueError("subgrid averaging requires at least two payloads")
    if labels is None:
        labels = tuple(f"source_{index}" for index in range(len(payloads)))
    if len(labels) != len(payloads):
        raise ValueError("labels and payloads must have the same length")

    analyses = [analyze_dwave_phase_hessian_payload(payload) for payload in payloads]
    q_reference = np.asarray(analyses[0].q_model, dtype=float)
    delta_reference = float(analyses[0].delta0_eV)
    for analysis in analyses[1:]:
        if not np.allclose(
            np.asarray(analysis.q_model, dtype=float),
            q_reference,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("all subgrid payloads must use the same q vector")
        if not np.isclose(
            float(analysis.delta0_eV),
            delta_reference,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("all subgrid payloads must use the same delta0_eV")

    sides: dict[str, dict[str, tuple[complex, complex]]] = {}
    for side_name in ("left", "right"):
        side_values = [getattr(analysis, side_name) for analysis in analyses]
        sides[side_name] = {
            "em_collective_contraction": (
                0.0 + 0.0j,
                _mean_complex([side.em_collective_phase for side in side_values]),
            ),
            "phase_rotation_bubble": (
                0.0 + 0.0j,
                _mean_complex([side.phase_rotation_bubble for side in side_values]),
            ),
            "phase_rotation_counterterm": (
                0.0 + 0.0j,
                _mean_complex([side.phase_rotation_counterterm for side in side_values]),
            ),
        }

    w_phase = complex(-2j * delta_reference)
    source_metadata = []
    for label, payload in zip(labels, payloads, strict=True):
        metadata = payload.get("metadata", {})
        grid = payload.get("grid", {})
        source_metadata.append(
            {
                "label": str(label),
                "schema": payload.get("schema"),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
                "grid": dict(grid) if isinstance(grid, Mapping) else {},
            }
        )

    return {
        "schema": "dwave_static_commensurate_phase_column_audit_v1",
        "audit": {
            "q_model": (float(q_reference[0]), float(q_reference[1])),
            "q_norm": float(np.linalg.norm(q_reference)),
            "delta0_eV": delta_reference,
            "w_left": (0.0 + 0.0j, w_phase),
            "w_right": (0.0 + 0.0j, w_phase),
            "component_sources": {
                side_name: {"collective_defect_parts": parts}
                for side_name, parts in sides.items()
            },
        },
        "primitive_metadata": {},
        "metadata": {
            "componentwise_subgrid_average": True,
            "average_formed_before_required_multiplier": True,
            "source_count": len(payloads),
            "sources": source_metadata,
        },
        "status": {
            "diagnostic_run_completed": True,
            "reduced_phase_column_only": True,
            "subgrid_averaged": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }


__all__ = ["average_dwave_phase_column_payloads"]
