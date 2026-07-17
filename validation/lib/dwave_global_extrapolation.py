"""Global complete-periodic convergence and small-frequency extrapolation helpers.

These helpers deliberately avoid local masks or nonuniform refinement.  Static
reference candidates are obtained from one fixed periodic-grid rule as ``nk`` is
increased.  Small-positive-frequency fits operate on complete periodic grids and
extrapolate real local-LT kernel channels to ``xi -> 0+``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from lno327.electrodynamics.basis import xy_to_lt_rotation


@dataclass(frozen=True)
class ExtrapolationSummary:
    """Robust summary of a family of fixed-form intercept fits."""

    estimate: float
    minimum: float
    maximum: float
    relative_spread: float
    best_model: str
    best_tail_points: int
    best_normalized_rms: float
    num_accepted_models: int


def relative_difference(value: float, reference: float) -> float:
    """Return an absolute relative difference with a safe denominator."""

    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _linear_intercept_fit(
    x_columns: np.ndarray,
    values: np.ndarray,
    *,
    model: str,
    tail_points: int,
) -> dict[str, Any]:
    y = np.asarray(values, dtype=float)
    columns = np.asarray(x_columns, dtype=float)
    if y.ndim != 1 or columns.ndim != 2 or columns.shape[0] != y.size:
        raise ValueError("fit arrays are not aligned")
    if not np.isfinite(y).all() or not np.isfinite(columns).all():
        raise ValueError("fit arrays must be finite")
    design = np.column_stack([np.ones(y.size, dtype=float), columns])
    if y.size <= design.shape[1]:
        raise ValueError("fit requires at least one residual degree of freedom")
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) != design.shape[1]:
        raise ValueError("fit design is rank deficient")
    predicted = design @ coefficients
    rms = float(np.sqrt(np.mean((predicted - y) ** 2)))
    scale = max(float(np.linalg.norm(y) / np.sqrt(y.size)), abs(float(coefficients[0])), 1e-30)
    return {
        "model": str(model),
        "tail_points": int(tail_points),
        "intercept": float(coefficients[0]),
        "normalized_rms": rms / scale,
        "rms": rms,
        "coefficients": [float(value) for value in coefficients],
    }


def static_power_law_fits(
    nks: Sequence[int],
    values: Sequence[float],
    *,
    powers: Sequence[int] = (1, 2, 3, 4),
    tail_sizes: Sequence[int] = (3, 4, 5),
) -> list[dict[str, Any]]:
    """Fit ``y(nk)=y_inf+a/nk**p`` over several fixed powers and tails."""

    nk = np.asarray(nks, dtype=float)
    y = np.asarray(values, dtype=float)
    if nk.ndim != 1 or y.shape != nk.shape or nk.size < 3:
        raise ValueError("static extrapolation requires at least three aligned points")
    if np.any(nk <= 0.0) or not np.isfinite(nk).all() or not np.isfinite(y).all():
        raise ValueError("static extrapolation inputs must be finite with positive nk")
    order = np.argsort(nk)
    nk, y = nk[order], y[order]
    fits: list[dict[str, Any]] = []
    for tail in sorted(set(int(value) for value in tail_sizes)):
        if tail < 3 or tail > nk.size:
            continue
        n_tail, y_tail = nk[-tail:], y[-tail:]
        for power in sorted(set(int(value) for value in powers)):
            if power <= 0:
                continue
            x = (n_tail / float(np.max(n_tail))) ** (-power)
            row = _linear_intercept_fit(
                x[:, None], y_tail, model=f"nk^-{power}", tail_points=tail
            )
            row["power"] = power
            row["nk_min"] = int(np.min(n_tail))
            row["nk_max"] = int(np.max(n_tail))
            fits.append(row)
    if not fits:
        raise ValueError("no valid static extrapolation fits were generated")
    return fits


def small_xi_fits(
    xi_eV: Sequence[float],
    values: Sequence[float],
    *,
    tail_sizes: Sequence[int] = (4, 5, 6, 8),
) -> list[dict[str, Any]]:
    """Fit several low-frequency intercept models on the smallest positive ``xi``.

    The candidates are linear in ``xi``, even-quadratic in ``xi**2``, and
    even-quartic in ``(xi**2, xi**4)``.  Reporting all three makes a possible
    nonanalytic or odd-in-frequency contamination visible instead of assuming an
    even expansion a priori.
    """

    xi = np.asarray(xi_eV, dtype=float)
    y = np.asarray(values, dtype=float)
    if xi.ndim != 1 or y.shape != xi.shape or xi.size < 3:
        raise ValueError("small-xi extrapolation requires aligned positive-frequency data")
    if np.any(xi <= 0.0) or not np.isfinite(xi).all() or not np.isfinite(y).all():
        raise ValueError("small-xi inputs must be finite and strictly positive")
    order = np.argsort(xi)
    xi, y = xi[order], y[order]
    fits: list[dict[str, Any]] = []
    for tail in sorted(set(int(value) for value in tail_sizes)):
        if tail < 3 or tail > xi.size:
            continue
        x_tail, y_tail = xi[:tail], y[:tail]
        scale = float(np.max(x_tail))
        z = x_tail / scale
        candidates = [
            ("linear_xi", z[:, None]),
            ("even_xi2", (z * z)[:, None]),
        ]
        if tail >= 4:
            candidates.append(("even_xi2_xi4", np.column_stack([z * z, z**4])))
        for model, columns in candidates:
            try:
                row = _linear_intercept_fit(
                    columns, y_tail, model=model, tail_points=tail
                )
            except ValueError:
                continue
            row["xi_min_eV"] = float(np.min(x_tail))
            row["xi_max_eV"] = float(np.max(x_tail))
            fits.append(row)
    if not fits:
        raise ValueError("no valid small-xi extrapolation fits were generated")
    return fits


def summarize_fit_ensemble(
    fits: Sequence[Mapping[str, Any]],
    *,
    rms_factor: float = 4.0,
    absolute_rms_floor: float = 1e-12,
) -> ExtrapolationSummary:
    """Summarize near-best fixed-form fits without trusting one ansatz alone."""

    rows = [dict(row) for row in fits]
    if not rows:
        raise ValueError("fit ensemble must be nonempty")
    finite = [
        row
        for row in rows
        if np.isfinite(float(row["intercept"]))
        and np.isfinite(float(row["normalized_rms"]))
    ]
    if not finite:
        raise ValueError("fit ensemble contains no finite rows")
    best = min(finite, key=lambda row: float(row["normalized_rms"]))
    threshold = max(
        float(absolute_rms_floor),
        float(rms_factor) * float(best["normalized_rms"]),
    )
    accepted = [row for row in finite if float(row["normalized_rms"]) <= threshold]
    intercepts = np.asarray([float(row["intercept"]) for row in accepted], dtype=float)
    estimate = float(np.median(intercepts))
    minimum = float(np.min(intercepts))
    maximum = float(np.max(intercepts))
    spread = (maximum - minimum) / max(abs(estimate), 1e-30)
    return ExtrapolationSummary(
        estimate=estimate,
        minimum=minimum,
        maximum=maximum,
        relative_spread=float(spread),
        best_model=str(best["model"]),
        best_tail_points=int(best["tail_points"]),
        best_normalized_rms=float(best["normalized_rms"]),
        num_accepted_models=len(accepted),
    )


def local_lt_kernel_proxies(kernel: object, q_model: Sequence[float]) -> dict[str, float]:
    """Return real static-channel proxies from one finite-frequency kernel.

    These are continuity diagnostics only: at ``xi>0`` they are not interpreted
    as a zero-mode sheet response and are never divided by frequency.
    """

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_model must be a finite nonzero two-vector")
    rotation = xy_to_lt_rotation(float(q[0]), float(q[1]))
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = rotation
    matrix = transform @ np.asarray(getattr(kernel, "k_eff"), dtype=complex) @ transform.T
    scale = max(float(np.linalg.norm(matrix.real)), 1.0)
    return {
        "chi_bar_proxy": float((-matrix[0, 0]).real),
        "dbar_t_proxy": float((-matrix[2, 2]).real),
        "k00_real": float(matrix[0, 0].real),
        "k00_imag": float(matrix[0, 0].imag),
        "ktt_real": float(matrix[2, 2].real),
        "ktt_imag": float(matrix[2, 2].imag),
        "kernel_lt_imaginary_relative_norm": float(np.linalg.norm(matrix.imag) / scale),
        "kernel_lt_frobenius_norm": float(np.linalg.norm(matrix)),
    }


def reference_estimate_from_json(path: str | Path) -> dict[str, float]:
    """Load the fail-closed static reference estimate written by the new runner."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reference = payload.get("reference_estimate")
    if not isinstance(reference, dict):
        raise ValueError("reference JSON does not contain 'reference_estimate'")
    result = {
        "chi_bar": float(reference["chi_bar"]),
        "dbar_t": float(reference["dbar_t"]),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("reference estimate contains non-finite values")
    return result


__all__ = [
    "ExtrapolationSummary",
    "local_lt_kernel_proxies",
    "reference_estimate_from_json",
    "relative_difference",
    "small_xi_fits",
    "static_power_law_fits",
    "summarize_fit_ensemble",
]
