"""Band-pair and shifted-Fermi-surface diagnostics for d-wave shift sensitivity.

The helpers in this module classify the exact-static BdG transitions sampled by
complete periodic shift rules.  They never modify the quadrature.  Bubble
transition strengths are retained by sorted BdG pair ``(m,n)``; particle/hole
weights and parent normal-state bands are reconstructed only for diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.response.finite_q_bdg import bdg_eigensystem_from_model_pairing
from lno327.response.finite_q_optimized import FiniteQQWorkspace, _vectorized_kubo_factors


PAIR_BLOCKS = ("k_ss", "k_seta", "k_etas", "k_etaeta", "equal_forward")


def _normal_band_content(
    spec: object,
    state: np.ndarray,
    kx: float,
    ky: float,
) -> tuple[float, int, np.ndarray]:
    """Return particle weight, dominant normal band, and two-band weights.

    The Nambu convention is ``(c_k, c^dagger_-k)``.  Particle amplitudes are
    projected with ``U_k^dagger`` and hole amplitudes with ``U_-k^T``.
    """

    vector = np.asarray(state, dtype=complex)
    dim = vector.size // 2
    if vector.shape != (2 * dim,):
        raise ValueError("BdG state must be a one-dimensional even-length vector")
    u, v = vector[:dim], vector[dim:]
    particle_weight = float(np.vdot(u, u).real)
    _, u_particle = np.linalg.eigh(np.asarray(spec.normal_hamiltonian(kx, ky), dtype=complex))
    _, u_hole = np.linalg.eigh(np.asarray(spec.normal_hamiltonian(-kx, -ky), dtype=complex))
    particle_band = np.abs(u_particle.conjugate().T @ u) ** 2
    hole_band = np.abs(u_hole.T @ v) ** 2
    weights = np.asarray(particle_band + hole_band, dtype=float)
    return particle_weight, int(np.argmax(weights)), weights


def _pair_strength(
    weighted_factor: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Frobenius strength of each rank-one channel-pair contribution."""

    left_norm = np.linalg.norm(np.asarray(left, dtype=complex), axis=1)
    right_norm = np.linalg.norm(np.asarray(right, dtype=complex), axis=1)
    return np.abs(weighted_factor) * left_norm * right_norm


def pointwise_bandpair_data(workspace: FiniteQQWorkspace) -> dict[str, Any]:
    """Return per-cell transition strengths and BdG/normal-band classifications."""

    material = workspace.material
    nk, nb = workspace.nk, workspace.nb
    factors = np.asarray(
        _vectorized_kubo_factors(workspace, np.asarray([0.0], dtype=float))[0],
        dtype=complex,
    )
    weighted = 0.5 * np.asarray(material.k_weights, dtype=float)[:, None, None] * factors
    left = np.asarray(workspace.left_vertices_band, dtype=complex)
    right = np.asarray(workspace.right_vertices_band, dtype=complex)

    strengths = {
        "k_ss": _pair_strength(weighted, left[:, :3], right[:, :3]),
        "k_seta": _pair_strength(weighted, left[:, :3], right[:, 3:5]),
        "k_etas": _pair_strength(weighted, left[:, 3:5], right[:, :3]),
        "k_etaeta": _pair_strength(weighted, left[:, 3:5], right[:, 3:5]),
    }
    occupation_difference = (
        np.asarray(workspace.occupations_minus, dtype=float)[:, :, None]
        - np.asarray(workspace.occupations_plus, dtype=float)[:, None, :]
    )
    rho = np.asarray(right[:, 0], dtype=complex)
    source_norm = np.linalg.norm(right[:, :3], axis=1)
    strengths["equal_forward"] = (
        0.5
        * np.asarray(material.k_weights, dtype=float)[:, None, None]
        * np.abs(occupation_difference)
        * np.abs(rho)
        * source_norm
    )

    particle_minus = np.zeros((nk, nb), dtype=float)
    particle_plus = np.zeros((nk, nb), dtype=float)
    normal_minus = np.zeros((nk, nb), dtype=int)
    normal_plus = np.zeros((nk, nb), dtype=int)
    normal_weights_minus = np.zeros((nk, nb, 2), dtype=float)
    normal_weights_plus = np.zeros((nk, nb, 2), dtype=float)
    qx, qy = map(float, workspace.q_model)
    spec, ansatz, amp = material.spec, material.ansatz, material.pairing_params

    for index, point in enumerate(np.asarray(material.k_points, dtype=float)):
        kx, ky = map(float, point)
        coordinates = ((kx - 0.5 * qx, ky - 0.5 * qy), (kx + 0.5 * qx, ky + 0.5 * qy))
        for side, (px, py) in enumerate(coordinates):
            pairing = ansatz.mean_pairing(px, py, amp)
            bands = bdg_eigensystem_from_model_pairing(spec, px, py, pairing)
            particle_target = particle_minus if side == 0 else particle_plus
            normal_target = normal_minus if side == 0 else normal_plus
            weights_target = normal_weights_minus if side == 0 else normal_weights_plus
            for band in range(nb):
                pweight, normal_band, band_weights = _normal_band_content(
                    spec, np.asarray(bands.states[:, band]), px, py
                )
                particle_target[index, band] = pweight
                normal_target[index, band] = normal_band
                weights_target[index, band] = band_weights

    energies_minus = np.asarray(workspace.energies_minus, dtype=float)
    energies_plus = np.asarray(workspace.energies_plus, dtype=float)
    pair_sign_crossing = energies_minus[:, :, None] * energies_plus[:, None, :] < 0.0
    pair_same_normal_band = normal_minus[:, :, None] == normal_plus[:, None, :]
    pair_same_bdg_index = np.broadcast_to(
        np.eye(nb, dtype=bool)[None, :, :], (nk, nb, nb)
    ).copy()
    pair_particle_minus = np.broadcast_to(particle_minus[:, :, None], (nk, nb, nb))
    pair_particle_plus = np.broadcast_to(particle_plus[:, None, :], (nk, nb, nb))
    pair_normal_minus = np.broadcast_to(normal_minus[:, :, None], (nk, nb, nb))
    pair_normal_plus = np.broadcast_to(normal_plus[:, None, :], (nk, nb, nb))

    return {
        "strengths": strengths,
        "energies_minus": energies_minus,
        "energies_plus": energies_plus,
        "particle_weight_minus": particle_minus,
        "particle_weight_plus": particle_plus,
        "normal_band_minus": normal_minus,
        "normal_band_plus": normal_plus,
        "normal_band_weights_minus": normal_weights_minus,
        "normal_band_weights_plus": normal_weights_plus,
        "pair_sign_crossing": pair_sign_crossing,
        "pair_same_normal_band": pair_same_normal_band,
        "pair_same_bdg_index": pair_same_bdg_index,
        "pair_particle_weight_minus": pair_particle_minus,
        "pair_particle_weight_plus": pair_particle_plus,
        "pair_normal_band_minus": pair_normal_minus,
        "pair_normal_band_plus": pair_normal_plus,
    }


def aggregate_rule_pair_strengths(
    shifts: Sequence[np.ndarray],
    weights: Sequence[float],
    cache: Mapping[tuple[float, float], Mapping[str, Any]],
    *,
    key_function,
) -> dict[str, np.ndarray]:
    """Weighted mean pair strengths for one complete-periodic shift rule."""

    weight_array = np.asarray(weights, dtype=float)
    if len(shifts) == 0 or weight_array.shape != (len(shifts),):
        raise ValueError("shifts and weights must be nonempty and aligned")
    weight_array = weight_array / float(np.sum(weight_array))
    result: dict[str, np.ndarray] = {}
    for block in PAIR_BLOCKS:
        stacked = np.stack(
            [np.asarray(cache[key_function(shift)]["strengths"][block], dtype=float) for shift in shifts],
            axis=0,
        )
        result[block] = np.tensordot(weight_array, stacked, axes=(0, 0))
    return result


def pair_strength_contrast(
    rule_a: Mapping[str, np.ndarray],
    rule_b: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Absolute rule contrast in transition strength for every sorted BdG pair."""

    if set(rule_a) != set(rule_b):
        raise ValueError("rule pair-strength dictionaries must expose the same blocks")
    return {
        name: np.abs(np.asarray(rule_b[name], dtype=float) - np.asarray(rule_a[name], dtype=float))
        for name in rule_a
    }


def aggregate_pair_classification(
    shifts_a: Sequence[np.ndarray],
    weights_a: Sequence[float],
    shifts_b: Sequence[np.ndarray],
    weights_b: Sequence[float],
    cache: Mapping[tuple[float, float], Mapping[str, Any]],
    *,
    key_function,
) -> dict[str, np.ndarray]:
    """Average pair classifications, giving each rule one-half of total weight."""

    events: list[tuple[float, Mapping[str, Any]]] = []
    for shifts, weights in ((shifts_a, weights_a), (shifts_b, weights_b)):
        normalized = np.asarray(weights, dtype=float)
        normalized = normalized / float(np.sum(normalized))
        events.extend(
            (0.5 * float(weight), cache[key_function(shift)])
            for shift, weight in zip(shifts, normalized, strict=True)
        )

    reference = events[0][1]
    shape = np.asarray(reference["pair_sign_crossing"]).shape
    result = {
        "sign_crossing_fraction": np.zeros(shape, dtype=float),
        "same_normal_band_fraction": np.zeros(shape, dtype=float),
        "same_bdg_index_fraction": np.zeros(shape, dtype=float),
        "particle_weight_minus": np.zeros(shape, dtype=float),
        "particle_weight_plus": np.zeros(shape, dtype=float),
        "normal_00_fraction": np.zeros(shape, dtype=float),
        "normal_11_fraction": np.zeros(shape, dtype=float),
        "normal_interband_fraction": np.zeros(shape, dtype=float),
        "ph_pp_fraction": np.zeros(shape, dtype=float),
        "ph_ph_fraction": np.zeros(shape, dtype=float),
        "ph_hp_fraction": np.zeros(shape, dtype=float),
        "ph_hh_fraction": np.zeros(shape, dtype=float),
    }
    for weight, event in events:
        crossing = np.asarray(event["pair_sign_crossing"], dtype=float)
        same_normal = np.asarray(event["pair_same_normal_band"], dtype=float)
        same_bdg = np.asarray(event["pair_same_bdg_index"], dtype=float)
        pminus = np.asarray(event["pair_particle_weight_minus"], dtype=float)
        pplus = np.asarray(event["pair_particle_weight_plus"], dtype=float)
        nminus = np.asarray(event["pair_normal_band_minus"], dtype=int)
        nplus = np.asarray(event["pair_normal_band_plus"], dtype=int)
        minus_particle = pminus >= 0.5
        plus_particle = pplus >= 0.5
        result["sign_crossing_fraction"] += weight * crossing
        result["same_normal_band_fraction"] += weight * same_normal
        result["same_bdg_index_fraction"] += weight * same_bdg
        result["particle_weight_minus"] += weight * pminus
        result["particle_weight_plus"] += weight * pplus
        result["normal_00_fraction"] += weight * ((nminus == 0) & (nplus == 0))
        result["normal_11_fraction"] += weight * ((nminus == 1) & (nplus == 1))
        result["normal_interband_fraction"] += weight * (nminus != nplus)
        result["ph_pp_fraction"] += weight * (minus_particle & plus_particle)
        result["ph_ph_fraction"] += weight * (minus_particle & ~plus_particle)
        result["ph_hp_fraction"] += weight * (~minus_particle & plus_particle)
        result["ph_hh_fraction"] += weight * (~minus_particle & ~plus_particle)
    return result


def dominant_pair_fields(
    contrast: Mapping[str, np.ndarray],
    classification: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Gather pair classifications at the maximum-contrast pair of each cell."""

    result: dict[str, dict[str, np.ndarray]] = {}
    for block, values in contrast.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 3 or array.shape[1] != array.shape[2]:
            raise ValueError("pair contrast arrays must have shape (nk, nb, nb)")
        nk, nb, _ = array.shape
        flat = np.argmax(array.reshape(nk, nb * nb), axis=1)
        m = flat // nb
        n = flat % nb
        rows = np.arange(nk)
        fields = {
            "m": m.astype(int),
            "n": n.astype(int),
            "contrast_mass": array[rows, m, n],
        }
        for name, source in classification.items():
            fields[name] = np.asarray(source)[rows, m, n]
        result[block] = fields
    return result


def normal_shifted_fs_fields(
    spec: object,
    centers: np.ndarray,
    q_model: Sequence[float],
) -> dict[str, np.ndarray]:
    """Normal-state band energies at base-cell centers shifted by +/-q/2."""

    points = np.asarray(centers, dtype=float)
    q = np.asarray(q_model, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or q.shape != (2,):
        raise ValueError("centers and q_model have incompatible shapes")
    minus = np.stack(
        [np.linalg.eigvalsh(np.asarray(spec.normal_hamiltonian(*(point - 0.5 * q)))) for point in points]
    )
    plus = np.stack(
        [np.linalg.eigvalsh(np.asarray(spec.normal_hamiltonian(*(point + 0.5 * q)))) for point in points]
    )
    same_band_crossing = np.any(minus * plus < 0.0, axis=1)
    return {
        "normal_minus_eV": np.asarray(minus, dtype=float),
        "normal_plus_eV": np.asarray(plus, dtype=float),
        "same_normal_band_sign_crossing": same_band_crossing,
        "minimum_shifted_normal_abs_eV": np.minimum(
            np.min(np.abs(minus), axis=1), np.min(np.abs(plus), axis=1)
        ),
    }


def _mass_fraction(mass: np.ndarray, condition: np.ndarray) -> float:
    values = np.asarray(mass, dtype=float)
    total = float(np.sum(values))
    return float(np.sum(values[np.asarray(condition, dtype=bool)]) / total) if total > 0.0 else float("nan")


def bandpair_mass_summary(
    masses: Mapping[str, np.ndarray],
    dominant: Mapping[str, Mapping[str, np.ndarray]],
    normal_fs: Mapping[str, np.ndarray],
    *,
    top_area_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """Summarize how much primitive difference mass belongs to each pair class."""

    rows: list[dict[str, Any]] = []
    for block in ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs"):
        pair_block = "equal_forward" if block == "ward_rhs" else block
        fields = dominant[pair_block]
        mass = np.asarray(masses[block], dtype=float)
        count = max(1, int(np.ceil(float(top_area_fraction) * len(mass))))
        top = np.argsort(mass)[::-1][:count]
        pair_codes = np.asarray(fields["m"], dtype=int) * 100 + np.asarray(fields["n"], dtype=int)
        pair_coverage: dict[str, float] = defaultdict(float)
        total = float(np.sum(mass))
        if total > 0.0:
            for code in np.unique(pair_codes):
                pair_coverage[f"{code // 100}->{code % 100}"] = float(
                    np.sum(mass[pair_codes == code]) / total
                )
        top_pair, top_pair_fraction = max(pair_coverage.items(), key=lambda item: item[1])
        crossing = np.asarray(fields["sign_crossing_fraction"], dtype=float) >= 0.5
        same_normal = np.asarray(fields["same_normal_band_fraction"], dtype=float) >= 0.5
        same_bdg = np.asarray(fields["same_bdg_index_fraction"], dtype=float) >= 0.5
        normal00 = np.asarray(fields["normal_00_fraction"], dtype=float) >= 0.5
        normal11 = np.asarray(fields["normal_11_fraction"], dtype=float) >= 0.5
        normal_inter = np.asarray(fields["normal_interband_fraction"], dtype=float) >= 0.5
        center_strip = np.asarray(normal_fs["same_normal_band_sign_crossing"], dtype=bool)
        rows.append(
            {
                "block": block,
                "top_pair": top_pair,
                "top_pair_mass_fraction": top_pair_fraction,
                "sign_crossing_mass_fraction": _mass_fraction(mass, crossing),
                "same_normal_band_mass_fraction": _mass_fraction(mass, same_normal),
                "same_bdg_index_mass_fraction": _mass_fraction(mass, same_bdg),
                "normal_00_mass_fraction": _mass_fraction(mass, normal00),
                "normal_11_mass_fraction": _mass_fraction(mass, normal11),
                "normal_interband_mass_fraction": _mass_fraction(mass, normal_inter),
                "center_shifted_fs_strip_mass_fraction": _mass_fraction(mass, center_strip),
                "top_area_fraction": float(count / len(mass)),
                "top_area_mass_captured": float(np.sum(mass[top]) / total) if total > 0.0 else float("nan"),
                "top_sign_crossing_fraction": float(np.mean(crossing[top])),
                "top_same_normal_band_fraction": float(np.mean(same_normal[top])),
                "top_center_shifted_fs_strip_fraction": float(np.mean(center_strip[top])),
                "top_normal_00_fraction": float(np.mean(normal00[top])),
                "top_normal_11_fraction": float(np.mean(normal11[top])),
                "top_normal_interband_fraction": float(np.mean(normal_inter[top])),
            }
        )
    return rows


__all__ = [
    "PAIR_BLOCKS",
    "aggregate_pair_classification",
    "aggregate_rule_pair_strengths",
    "bandpair_mass_summary",
    "dominant_pair_fields",
    "normal_shifted_fs_fields",
    "pair_strength_contrast",
    "pointwise_bandpair_data",
]
