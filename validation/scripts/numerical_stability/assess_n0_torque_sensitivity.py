#!/usr/bin/env python3
"""Assess integrand-level n=0 torque sensitivity against an n>=1 baseline.

This is not a formal Casimir calculation. It compares one fixed
``k_parallel, phi, theta`` integrand-level partial Matsubara sum against an
extrapolated n=0 proxy so the local baseline can decide whether ``skip`` is a
defensible conservative default.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    CasimirSetup,
    ConductivityTensor,
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    casimir_torque_integrand,
    k_weights,
    local_response_imag_axis,
    local_response_matsubara_index,
    matrix_symmetry_diagnostics,
    model_response_to_sheet_conductivity,
    uniform_bz_mesh,
)
from lno327.casimir import matsubara_frequency  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
POLICIES = ("skip", "extrapolate_from_lowest_matsubara", "use_static_kernel")
ZERO_TORQUE_TOLERANCE = 1e-30
RATIO_EPS = 1e-300

REQUIRED_NPZ_FIELDS = {
    "kind",
    "policy",
    "status",
    "approximate",
    "not_used_as_sigma",
    "tau_n0_proxy",
    "tau_ref_n_ge1",
    "ratio_abs_n0_to_ref",
    "sensitivity_threshold",
    "skip_acceptability",
    "n0_sensitivity",
    "delta_n0_proxy",
    "relative_offdiag_n0_proxy",
    "static_kernel_xx",
    "static_kernel_yy",
    "static_delta",
    "static_relative_offdiag",
    "notes",
    "theta_scan",
    "tau_ref_theta",
    "tau_n0_proxy_theta",
    "ratio_theta",
}


def _complex_nan() -> complex:
    return complex(np.nan, np.nan)


def _tensor_from_matrix(matrix: np.ndarray) -> ConductivityTensor:
    sheet = model_response_to_sheet_conductivity(matrix)
    sheet_matrix = sheet.tensor.matrix()
    return ConductivityTensor(
        xx=sheet_matrix[0, 0],
        yy=sheet_matrix[1, 1],
        xy=sheet_matrix[0, 1],
        yx=sheet_matrix[1, 0],
    )


def _toy_anisotropic_tensor() -> ConductivityTensor:
    return ConductivityTensor(xx=2.0e-4, yy=1.0e-4, xy=0.0, yx=0.0)


def _diagnostics(matrix: np.ndarray | None) -> tuple[complex, float]:
    if matrix is None:
        return _complex_nan(), np.nan
    values = matrix_symmetry_diagnostics(matrix)
    return complex(values["delta"]), float(values["relative_offdiag"])


def _classify_sensitivity(
    tau_ref: complex,
    tau_n0_proxy: complex,
    ratio: float,
    threshold: float,
) -> tuple[str, str]:
    ref_abs = abs(tau_ref)
    proxy_abs = abs(tau_n0_proxy)
    if ref_abs < ZERO_TORQUE_TOLERANCE and proxy_abs < ZERO_TORQUE_TOLERANCE:
        return "negligible_zero_baseline", "acceptable_for_current_local_baseline"
    if ref_abs < ZERO_TORQUE_TOLERANCE and proxy_abs >= ZERO_TORQUE_TOLERANCE:
        return "finite_n0_proxy_over_zero_reference", "not_acceptable_requires_zero_frequency_model"
    if ratio < threshold:
        return "below_threshold", "acceptable_for_current_local_baseline"
    return "above_threshold", "not_acceptable_requires_zero_frequency_model"


def _ratio(proxy: complex, reference: complex) -> float:
    ref_abs = abs(reference)
    proxy_abs = abs(proxy)
    if ref_abs < ZERO_TORQUE_TOLERANCE and proxy_abs < ZERO_TORQUE_TOLERANCE:
        return 0.0
    return float(proxy_abs / (ref_abs + RATIO_EPS))


def _ratio_array(proxy: np.ndarray, reference: np.ndarray) -> np.ndarray:
    proxy_abs = np.abs(proxy)
    ref_abs = np.abs(reference)
    ratios = proxy_abs / (ref_abs + RATIO_EPS)
    both_zero = (proxy_abs < ZERO_TORQUE_TOLERANCE) & (ref_abs < ZERO_TORQUE_TOLERANCE)
    ratios[both_zero] = 0.0
    return ratios


def _local_sheet_tensor(
    kind: str,
    n: int,
    mesh: np.ndarray,
    weights: np.ndarray,
    temperature_K: float,
    eta_eV: float,
    pairing_params: PairingAmplitudes,
) -> ConductivityTensor:
    omega_eV = bosonic_matsubara_energy_eV(n, temperature_K)
    response = local_response_imag_axis(
        kind,  # type: ignore[arg-type]
        omega_eV,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=pairing_params,
        k_weights=weights,
    )
    return _tensor_from_matrix(response.matrix)


def _partial_sum_torque(
    tensors_by_n: dict[int, ConductivityTensor],
    setup: CasimirSetup,
    temperature_K: float,
    k_parallel: float,
    phi: float,
    theta: float,
) -> complex:
    total = 0.0 + 0.0j
    for n, tensor in tensors_by_n.items():
        total += casimir_torque_integrand(
            setup,
            matsubara_frequency(n, temperature_K),
            k_parallel,
            phi,
            theta,
            tensor,
            tensor,
        )
    return total


def _partial_sum_theta(
    tensors_by_n: dict[int, ConductivityTensor],
    setup: CasimirSetup,
    temperature_K: float,
    k_parallel: float,
    phi: float,
    theta_values: np.ndarray,
) -> np.ndarray:
    values = np.empty(theta_values.size, dtype=complex)
    for index, theta in enumerate(theta_values):
        values[index] = _partial_sum_torque(
            tensors_by_n,
            setup,
            temperature_K,
            k_parallel,
            phi,
            float(theta),
        )
    return values


def _single_theta(
    tensor: ConductivityTensor,
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    theta_values: np.ndarray,
) -> np.ndarray:
    values = np.empty(theta_values.size, dtype=complex)
    for index, theta in enumerate(theta_values):
        values[index] = casimir_torque_integrand(
            setup,
            xi,
            k_parallel,
            phi,
            float(theta),
            tensor,
            tensor,
        )
    return values


def _kind_baseline(
    kind: str,
    nk: int,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    reference_min: int,
    reference_max: int,
    setup: CasimirSetup,
    k_parallel: float,
    phi: float,
    theta: float,
    theta_values: np.ndarray,
) -> dict[str, object]:
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    params = PairingAmplitudes(delta0_eV=delta0_eV)
    tensors_by_n = {
        n: _local_sheet_tensor(kind, n, mesh, weights, temperature_K, eta_eV, params)
        for n in range(reference_min, reference_max + 1)
    }
    tau_ref = _partial_sum_torque(
        tensors_by_n,
        setup,
        temperature_K,
        k_parallel,
        phi,
        theta,
    )
    tau_ref_theta = _partial_sum_theta(
        tensors_by_n,
        setup,
        temperature_K,
        k_parallel,
        phi,
        theta_values,
    )

    proxy_result = local_response_matsubara_index(
        kind,  # type: ignore[arg-type]
        0,
        temperature_K,
        policy="extrapolate_from_lowest_matsubara",
        nk=nk,
        eta_eV=eta_eV,
        pairing_params=params,
    )
    proxy_matrix = np.asarray(proxy_result.matrix, dtype=complex)
    proxy_tensor = _tensor_from_matrix(proxy_matrix)
    tau_proxy = casimir_torque_integrand(
        setup,
        0.0,
        k_parallel,
        phi,
        theta,
        proxy_tensor,
        proxy_tensor,
    )
    tau_proxy_theta = _single_theta(proxy_tensor, setup, 0.0, k_parallel, phi, theta_values)
    proxy_delta, proxy_offdiag = _diagnostics(proxy_matrix)

    static_result = local_response_matsubara_index(
        kind,  # type: ignore[arg-type]
        0,
        temperature_K,
        policy="use_static_kernel",
        nk=nk,
        eta_eV=eta_eV,
        pairing_params=params,
    )
    static_matrix = None if static_result.matrix is None else np.asarray(static_result.matrix, dtype=complex)
    static_delta, static_offdiag = _diagnostics(static_matrix)

    return {
        "tau_ref": tau_ref,
        "tau_ref_theta": tau_ref_theta,
        "tau_proxy": tau_proxy,
        "tau_proxy_theta": tau_proxy_theta,
        "proxy_delta": proxy_delta,
        "proxy_offdiag": proxy_offdiag,
        "static_matrix": static_matrix,
        "static_delta": static_delta,
        "static_offdiag": static_offdiag,
    }


def _toy_baseline(
    reference_min: int,
    reference_max: int,
    setup: CasimirSetup,
    temperature_K: float,
    k_parallel: float,
    phi: float,
    theta: float,
    theta_values: np.ndarray,
) -> dict[str, object]:
    tensor = _toy_anisotropic_tensor()
    tensors_by_n = {n: tensor for n in range(reference_min, reference_max + 1)}
    tau_ref = _partial_sum_torque(tensors_by_n, setup, temperature_K, k_parallel, phi, theta)
    tau_ref_theta = _partial_sum_theta(tensors_by_n, setup, temperature_K, k_parallel, phi, theta_values)
    tau_proxy = casimir_torque_integrand(setup, 0.0, k_parallel, phi, theta, tensor, tensor)
    tau_proxy_theta = _single_theta(tensor, setup, 0.0, k_parallel, phi, theta_values)
    proxy_matrix = np.array([[tensor.xx, tensor.xy], [tensor.yx, tensor.yy]], dtype=complex)
    proxy_delta, proxy_offdiag = _diagnostics(proxy_matrix)
    return {
        "tau_ref": tau_ref,
        "tau_ref_theta": tau_ref_theta,
        "tau_proxy": tau_proxy,
        "tau_proxy_theta": tau_proxy_theta,
        "proxy_delta": proxy_delta,
        "proxy_offdiag": proxy_offdiag,
        "static_matrix": None,
        "static_delta": _complex_nan(),
        "static_offdiag": np.nan,
    }


def assess_n0_sensitivity(
    kinds: list[str],
    nk: int,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    distance_m: float,
    k_parallel: float,
    phi: float,
    theta: float,
    reference_min: int,
    reference_max: int,
    sensitivity_threshold: float,
    theta_scan_min: float,
    theta_scan_max: float,
    theta_scan_num: int,
    include_toy_anisotropic_control: bool = False,
) -> dict[str, np.ndarray]:
    """Assess whether baseline n=0 skip is acceptable at integrand level."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    if reference_min < 1 or reference_max < reference_min:
        raise ValueError("reference Matsubara range must satisfy 1 <= min <= max")
    if sensitivity_threshold <= 0.0:
        raise ValueError("sensitivity threshold must be positive")
    if theta_scan_num < 2:
        raise ValueError("theta scan must contain at least two points")

    setup = CasimirSetup(temperature=temperature_K, distance=distance_m)
    theta_values = np.linspace(theta_scan_min, theta_scan_max, theta_scan_num)
    effective_kinds = list(kinds)
    if include_toy_anisotropic_control:
        effective_kinds.append("toy_anisotropic")
    rows = [(kind, policy) for kind in effective_kinds for policy in POLICIES]
    row_count = len(rows)

    data: dict[str, np.ndarray] = {
        "kind": np.empty(row_count, dtype="U24"),
        "policy": np.empty(row_count, dtype="U48"),
        "status": np.empty(row_count, dtype="U64"),
        "approximate": np.zeros(row_count, dtype=bool),
        "not_used_as_sigma": np.zeros(row_count, dtype=bool),
        "tau_n0_proxy": np.full(row_count, _complex_nan(), dtype=complex),
        "tau_ref_n_ge1": np.full(row_count, _complex_nan(), dtype=complex),
        "ratio_abs_n0_to_ref": np.full(row_count, np.nan, dtype=float),
        "sensitivity_threshold": np.full(row_count, sensitivity_threshold, dtype=float),
        "skip_acceptability": np.empty(row_count, dtype="U64"),
        "n0_sensitivity": np.empty(row_count, dtype="U64"),
        "delta_n0_proxy": np.full(row_count, _complex_nan(), dtype=complex),
        "relative_offdiag_n0_proxy": np.full(row_count, np.nan, dtype=float),
        "static_kernel_xx": np.full(row_count, _complex_nan(), dtype=complex),
        "static_kernel_yy": np.full(row_count, _complex_nan(), dtype=complex),
        "static_delta": np.full(row_count, _complex_nan(), dtype=complex),
        "static_relative_offdiag": np.full(row_count, np.nan, dtype=float),
        "notes": np.empty(row_count, dtype=object),
        "theta_scan": theta_values,
        "tau_ref_theta": np.full((row_count, theta_values.size), _complex_nan(), dtype=complex),
        "tau_n0_proxy_theta": np.full((row_count, theta_values.size), _complex_nan(), dtype=complex),
        "ratio_theta": np.full((row_count, theta_values.size), np.nan, dtype=float),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "delta0_eV": np.array(delta0_eV),
        "eta_eV": np.array(eta_eV),
        "distance_m": np.array(distance_m),
        "k_parallel": np.array(k_parallel),
        "phi": np.array(phi),
        "theta": np.array(theta),
        "reference_matsubara_min": np.array(reference_min),
        "reference_matsubara_max": np.array(reference_max),
        "include_toy_anisotropic_control": np.array(include_toy_anisotropic_control),
    }

    baselines: dict[str, dict[str, object]] = {}
    for kind in kinds:
        baselines[kind] = _kind_baseline(
            kind,
            nk,
            temperature_K,
            delta0_eV,
            eta_eV,
            reference_min,
            reference_max,
            setup,
            k_parallel,
            phi,
            theta,
            theta_values,
        )
    if include_toy_anisotropic_control:
        baselines["toy_anisotropic"] = _toy_baseline(
            reference_min,
            reference_max,
            setup,
            temperature_K,
            k_parallel,
            phi,
            theta,
            theta_values,
        )

    for index, (kind, policy) in enumerate(rows):
        baseline = baselines[kind]
        tau_ref = complex(baseline["tau_ref"])
        tau_proxy = complex(baseline["tau_proxy"])
        ratio = _ratio(tau_proxy, tau_ref)
        n0_sensitivity, skip_acceptability = _classify_sensitivity(
            tau_ref,
            tau_proxy,
            ratio,
            sensitivity_threshold,
        )
        tau_ref_theta = np.asarray(baseline["tau_ref_theta"], dtype=complex)
        tau_proxy_theta = np.asarray(baseline["tau_proxy_theta"], dtype=complex)

        data["kind"][index] = kind
        data["policy"][index] = policy
        data["tau_ref_n_ge1"][index] = tau_ref
        data["tau_ref_theta"][index] = tau_ref_theta
        data["sensitivity_threshold"][index] = sensitivity_threshold
        data["skip_acceptability"][index] = skip_acceptability
        data["n0_sensitivity"][index] = n0_sensitivity

        if policy == "skip":
            data["status"][index] = "baseline_skip_n0"
            data["approximate"][index] = False
            data["not_used_as_sigma"][index] = True
            data["ratio_abs_n0_to_ref"][index] = ratio
            data["tau_n0_proxy"][index] = tau_proxy
            data["tau_n0_proxy_theta"][index] = tau_proxy_theta
            data["ratio_theta"][index] = _ratio_array(tau_proxy_theta, tau_ref_theta)
            data["notes"][index] = (
                "current local baseline skips n=0",
                "acceptability is judged using extrapolated n=0 proxy sensitivity",
                "integrand-level partial Matsubara sum only",
            )
        elif policy == "extrapolate_from_lowest_matsubara":
            data["status"][index] = "n0_proxy_from_lowest_matsubara"
            data["approximate"][index] = True
            data["not_used_as_sigma"][index] = False
            data["tau_n0_proxy"][index] = tau_proxy
            data["ratio_abs_n0_to_ref"][index] = ratio
            data["tau_n0_proxy_theta"][index] = tau_proxy_theta
            data["ratio_theta"][index] = _ratio_array(tau_proxy_theta, tau_ref_theta)
            data["delta_n0_proxy"][index] = complex(baseline["proxy_delta"])
            data["relative_offdiag_n0_proxy"][index] = float(baseline["proxy_offdiag"])
            data["notes"][index] = (
                "sensitivity estimate only, not final n=0 physics",
                "may enter diagnostic reflection integrand after unit conversion",
            )
        else:
            static_matrix = baseline["static_matrix"]
            data["status"][index] = "static_kernel_diagnostic"
            data["approximate"][index] = False
            data["not_used_as_sigma"][index] = True
            if static_matrix is not None:
                matrix = np.asarray(static_matrix, dtype=complex)
                data["static_kernel_xx"][index] = matrix[0, 0]
                data["static_kernel_yy"][index] = matrix[1, 1]
            data["static_delta"][index] = complex(baseline["static_delta"])
            data["static_relative_offdiag"][index] = float(baseline["static_offdiag"])
            data["notes"][index] = (
                "stiffness-like static-kernel diagnostic only",
                "static kernel is not Sigma_SC(0)",
                "not used as sheet conductivity or reflection input",
            )

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "response" / "n0_sensitivity" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "response" / "n0_sensitivity" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_tau_proxy_vs_ref.png",
        figure_dir / f"{output_prefix.name}_ratio.png",
        figure_dir / f"{output_prefix.name}_theta_scan.png",
        figure_dir / f"{output_prefix.name}_static_kernel_anisotropy.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    npz_path, csv_path, tau_plot_path, ratio_plot_path, theta_plot_path, static_plot_path = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tau_plot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "kind",
        "policy",
        "status",
        "approximate",
        "not_used_as_sigma",
        "tau_n0_proxy",
        "tau_ref_n_ge1",
        "ratio_abs_n0_to_ref",
        "sensitivity_threshold",
        "skip_acceptability",
        "n0_sensitivity",
        "delta_n0_proxy",
        "relative_offdiag_n0_proxy",
        "static_kernel_xx",
        "static_kernel_yy",
        "static_delta",
        "static_relative_offdiag",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(kind) for kind in data["kind"]))
    x = np.arange(len(kinds), dtype=float)
    skip_indices = [int(np.flatnonzero((data["kind"] == kind) & (data["policy"] == "skip"))[0]) for kind in kinds]
    proxy_indices = [
        int(np.flatnonzero((data["kind"] == kind) & (data["policy"] == "extrapolate_from_lowest_matsubara"))[0])
        for kind in kinds
    ]
    static_indices = [
        int(np.flatnonzero((data["kind"] == kind) & (data["policy"] == "use_static_kernel"))[0])
        for kind in kinds
    ]

    fig_tau, ax_tau = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax_tau.bar(x - 0.18, [abs(data["tau_ref_n_ge1"][i]) for i in skip_indices], width=0.36, label="|tau_ref n>=1|")
    ax_tau.bar(x + 0.18, [abs(data["tau_n0_proxy"][i]) for i in proxy_indices], width=0.36, label="|tau_n0 proxy|")
    ax_tau.set_xticks(x, kinds)
    ax_tau.set_yscale("symlog", linthresh=1e-30)
    ax_tau.set_ylabel("torque integrand magnitude")
    ax_tau.set_title("n=0 proxy versus n>=1 partial-sum baseline")
    style_publication_axis(ax_tau)
    save_publication_figure(fig_tau, tau_plot_path)
    plt.close(fig_tau)

    fig_ratio, ax_ratio = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ratios = [data["ratio_abs_n0_to_ref"][i] for i in skip_indices]
    ax_ratio.bar(x, ratios, width=0.5)
    ax_ratio.axhline(float(data["sensitivity_threshold"][0]), color="black", linestyle="--", linewidth=1.0)
    ax_ratio.set_xticks(x, kinds)
    ax_ratio.set_yscale("symlog", linthresh=1e-6)
    ax_ratio.set_ylabel("|tau_n0 proxy| / |tau_ref|")
    ax_ratio.set_title("n=0 sensitivity ratio")
    style_publication_axis(ax_ratio, legend=False)
    save_publication_figure(fig_ratio, ratio_plot_path)
    plt.close(fig_ratio)

    fig_theta, ax_theta = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    theta_values = data["theta_scan"]
    for kind, index in zip(kinds, skip_indices, strict=True):
        ax_theta.plot(theta_values, data["tau_ref_theta"][index].real, label=f"{kind} ref")
        ax_theta.plot(theta_values, data["tau_n0_proxy_theta"][index].real, linestyle="--", label=f"{kind} n0 proxy")
    ax_theta.set_xlabel(r"$\theta$ (rad)")
    ax_theta.set_ylabel("Re torque integrand")
    ax_theta.set_title("theta scan: n>=1 baseline and n=0 proxy")
    style_publication_axis(ax_theta)
    save_publication_figure(fig_theta, theta_plot_path)
    plt.close(fig_theta)

    fig_static, ax_static = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax_static.plot(x, np.abs(data["static_delta"][static_indices]), marker="o", label="|static_delta|")
    ax_static.plot(x, data["static_relative_offdiag"][static_indices], marker="s", linestyle="--", label="static offdiag")
    ax_static.set_xticks(x, kinds)
    ax_static.set_yscale("symlog", linthresh=1e-16)
    ax_static.set_ylabel("relative diagnostic")
    ax_static.set_title("static-kernel anisotropy diagnostics")
    style_publication_axis(ax_static)
    save_publication_figure(fig_static, static_plot_path)
    plt.close(fig_static)

    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    for kind in dict.fromkeys(str(item) for item in data["kind"]):
        index = int(np.flatnonzero((data["kind"] == kind) & (data["policy"] == "skip"))[0])
        print(f"kind = {kind}")
        print(f"tau_ref_n_ge1 = {data['tau_ref_n_ge1'][index]}")
        print(f"tau_n0_proxy = {data['tau_n0_proxy'][index]}")
        print(f"ratio_abs_n0_to_ref = {float(data['ratio_abs_n0_to_ref'][index])}")
        print(f"n0_sensitivity = {data['n0_sensitivity'][index]}")
        print(f"skip_acceptability = {data['skip_acceptability'][index]}")
    print("note = integrand-level partial Matsubara-sum sensitivity only; not a formal Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--distance", type=float, default=3e-8)
    parser.add_argument("--k-parallel", type=float, default=1e6)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--reference-matsubara-min", type=int, default=1)
    parser.add_argument("--reference-matsubara-max", type=int, default=8)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.01)
    parser.add_argument("--theta-scan-min", type=float, default=0.0)
    parser.add_argument("--theta-scan-max", type=float, default=np.pi)
    parser.add_argument("--theta-scan-num", type=int, default=41)
    parser.add_argument("--include-toy-anisotropic-control", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "validation" / "outputs" / "archive" / "response" / "n0_sensitivity" / "data" / "n0_sensitivity",
    )
    args = parser.parse_args()

    data = assess_n0_sensitivity(
        args.kinds,
        args.nk,
        args.temperature,
        args.delta0,
        args.eta,
        args.distance,
        args.k_parallel,
        args.phi,
        args.theta,
        args.reference_matsubara_min,
        args.reference_matsubara_max,
        args.sensitivity_threshold,
        args.theta_scan_min,
        args.theta_scan_max,
        args.theta_scan_num,
        args.include_toy_anisotropic_control,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(f"figure_paths = {paths[2]}, {paths[3]}, {paths[4]}, {paths[5]}")


if __name__ == "__main__":
    main()
