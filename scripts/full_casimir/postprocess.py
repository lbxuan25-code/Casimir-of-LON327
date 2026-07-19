from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import hashlib
import json
import math
import re

from .config import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSTPROCESS_ROOT,
    DEFAULT_SCAN_STEP_DEG,
    DEFAULT_TARGET_MAX_DEG,
    DEFAULT_TARGET_MIN_DEG,
    PROFILE_NAME,
    inclusive_integer_grid,
)


@dataclass(frozen=True)
class EnergyPoint:
    pairing: str
    angle_deg: int
    case: str
    manifest_status: str
    status: str
    termination_reason: str
    matsubara_converged: bool
    energy_J_m2: float | None
    error_J_m2: float | None
    temperature_K: float
    separation_nm: float
    series_signature: str

    @property
    def usable(self) -> bool:
        return (
            self.manifest_status == "completed"
            and self.matsubara_converged
            and self.energy_J_m2 is not None
            and self.error_J_m2 is not None
            and math.isfinite(self.energy_J_m2)
            and math.isfinite(self.error_J_m2)
            and self.error_J_m2 >= 0.0
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _case_regex(profile: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(spm|dwave)_T[^_]+K_d[^_]+nm_theta_([mp])(\d{{3}})deg_"
        rf"{re.escape(profile)}$"
    )


def decode_angle(sign: str, magnitude: str) -> int:
    value = int(magnitude)
    return -value if sign == "m" else value


def _point_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        point = config["outer_tail_config"]["joint_config"]["radial_config"][
            "point_config"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("run config does not contain the canonical point configuration") from exc
    if not isinstance(point, Mapping):
        raise ValueError("point configuration must be an object")
    return point


def _series_signature(config: Mapping[str, Any]) -> str:
    """Hash result-changing policy while excluding angle and orchestration controls."""

    payload = json.loads(json.dumps(config))
    ignored = {
        "plate_angles_deg",
        "point_cache_path",
        "transverse_checkpoint_path",
        "workers",
        "parallel_mode",
        "memory_budget_gb",
        "max_context_workers",
        "certifier_q_batch_size",
    }

    def prune(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: prune(child)
                for key, child in value.items()
                if key not in ignored
            }
        if isinstance(value, list):
            return [prune(child) for child in value]
        return value

    encoded = json.dumps(
        prune(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_energy_points(
    *,
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    profile: str = PROFILE_NAME,
) -> list[EnergyPoint]:
    pattern = _case_regex(profile)
    output: list[EnergyPoint] = []
    seen: set[tuple[str, int]] = set()

    if not run_root.exists():
        return output

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = pattern.fullmatch(run_dir.name)
        if match is None:
            continue

        pairing = match.group(1)
        angle_deg = decode_angle(match.group(2), match.group(3))
        identity = (pairing, angle_deg)
        if identity in seen:
            raise ValueError(
                f"profile {profile!r} contains duplicate {pairing} angle {angle_deg}"
            )
        seen.add(identity)

        summary = _read_json(run_dir / "summary.json")
        manifest = _read_json(run_dir / "manifest.json")
        config = _read_json(run_dir / "config.json")
        point = _point_config(config)
        configured_pairings = tuple(str(value) for value in point.get("pairings", ()))
        configured_angles = point.get("plate_angles_deg", ())
        if configured_pairings != (pairing,):
            raise ValueError(
                f"case {run_dir.name} pairing does not match its configuration"
            )
        if (
            not isinstance(configured_angles, list)
            or len(configured_angles) != 2
            or not math.isclose(
                float(configured_angles[1]),
                float(angle_deg),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                f"case {run_dir.name} angle does not match its configuration"
            )
        temperature_K = float(point.get("temperature_K"))
        separation_nm = float(point.get("separation_nm"))
        if not math.isfinite(temperature_K) or temperature_K <= 0.0:
            raise ValueError(f"case {run_dir.name} has invalid temperature")
        if not math.isfinite(separation_nm) or separation_nm <= 0.0:
            raise ValueError(f"case {run_dir.name} has invalid separation")

        payload = summary.get("pairings", {}).get(pairing, {})
        output.append(
            EnergyPoint(
                pairing=pairing,
                angle_deg=angle_deg,
                case=run_dir.name,
                manifest_status=str(manifest.get("status", "missing")),
                status=str(summary.get("status", manifest.get("status", "missing"))),
                termination_reason=str(
                    summary.get(
                        "termination_reason",
                        manifest.get("termination_reason", manifest.get("error", "")),
                    )
                ),
                matsubara_converged=bool(summary.get("matsubara_converged", False)),
                energy_J_m2=_finite_float(
                    payload.get("finite_matsubara_partial_J_m2")
                ),
                error_J_m2=_finite_float(
                    payload.get("estimated_total_error_J_m2")
                ),
                temperature_K=temperature_K,
                separation_nm=separation_nm,
                series_signature=_series_signature(config),
            )
        )

    physical = {(point.temperature_K, point.separation_nm) for point in output}
    if len(physical) > 1:
        raise ValueError(
            f"profile {profile!r} mixes multiple temperatures or separations: {physical}"
        )
    for pairing in {point.pairing for point in output}:
        signatures = {
            point.series_signature for point in output if point.pairing == pairing
        }
        if len(signatures) > 1:
            raise ValueError(
                f"profile {profile!r} mixes incompatible numerical policies for {pairing}"
            )
    return output


def five_point_torque(
    energies: Mapping[int, float],
    *,
    angle_deg: int,
    step_deg: int,
) -> float:
    h_rad = math.radians(step_deg)
    required = (
        angle_deg - 2 * step_deg,
        angle_deg - step_deg,
        angle_deg + step_deg,
        angle_deg + 2 * step_deg,
    )
    missing = [value for value in required if value not in energies]
    if missing:
        raise KeyError(f"missing energy angles: {missing}")
    derivative = (
        energies[angle_deg - 2 * step_deg]
        - 8.0 * energies[angle_deg - step_deg]
        + 8.0 * energies[angle_deg + step_deg]
        - energies[angle_deg + 2 * step_deg]
    ) / (12.0 * h_rad)
    return -derivative


def three_point_torque(
    energies: Mapping[int, float],
    *,
    angle_deg: int,
    step_deg: int,
) -> float:
    h_rad = math.radians(step_deg)
    required = (angle_deg - step_deg, angle_deg + step_deg)
    missing = [value for value in required if value not in energies]
    if missing:
        raise KeyError(f"missing energy angles: {missing}")
    return -(
        energies[angle_deg + step_deg] - energies[angle_deg - step_deg]
    ) / (2.0 * h_rad)


def five_point_torque_error_bound(
    errors: Mapping[int, float],
    *,
    angle_deg: int,
    step_deg: int,
) -> float:
    h_rad = math.radians(step_deg)
    required = (
        angle_deg - 2 * step_deg,
        angle_deg - step_deg,
        angle_deg + step_deg,
        angle_deg + 2 * step_deg,
    )
    missing = [value for value in required if value not in errors]
    if missing:
        raise KeyError(f"missing error angles: {missing}")
    return (
        errors[angle_deg - 2 * step_deg]
        + 8.0 * errors[angle_deg - step_deg]
        + 8.0 * errors[angle_deg + step_deg]
        + errors[angle_deg + 2 * step_deg]
    ) / (12.0 * h_rad)


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _write_energy_csv(path: Path, points: Sequence[EnergyPoint]) -> None:
    fields = (
        "pairing",
        "angle_deg",
        "case",
        "manifest_status",
        "status",
        "termination_reason",
        "matsubara_converged",
        "temperature_K",
        "separation_nm",
        "energy_J_m2",
        "energy_error_bound_J_m2",
        "usable_for_torque",
    )
    rows = [
        {
            "pairing": point.pairing,
            "angle_deg": point.angle_deg,
            "case": point.case,
            "manifest_status": point.manifest_status,
            "status": point.status,
            "termination_reason": point.termination_reason,
            "matsubara_converged": point.matsubara_converged,
            "temperature_K": point.temperature_K,
            "separation_nm": point.separation_nm,
            "energy_J_m2": point.energy_J_m2,
            "energy_error_bound_J_m2": point.error_J_m2,
            "usable_for_torque": point.usable,
        }
        for point in sorted(points, key=lambda item: (item.pairing, item.angle_deg))
    ]
    _atomic_csv(path, fields, rows)


def postprocess_torque(
    *,
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_POSTPROCESS_ROOT,
    profile: str = PROFILE_NAME,
    step_deg: int = DEFAULT_SCAN_STEP_DEG,
    target_min_deg: int = DEFAULT_TARGET_MIN_DEG,
    target_max_deg: int = DEFAULT_TARGET_MAX_DEG,
) -> tuple[Path, Path, Path, bool]:
    points = collect_energy_points(run_root=run_root, profile=profile)
    destination = output_root / profile
    energy_csv = destination / "free_energy.csv"
    torque_csv = destination / "torque.csv"
    metadata_json = destination / "metadata.json"
    _write_energy_csv(energy_csv, points)

    by_pairing: dict[str, dict[int, EnergyPoint]] = {"spm": {}, "dwave": {}}
    for point in points:
        by_pairing.setdefault(point.pairing, {})[point.angle_deg] = point

    target_angles = inclusive_integer_grid(target_min_deg, target_max_deg, step_deg)
    rows: list[dict[str, Any]] = []
    all_available = True

    for pairing in ("spm", "dwave"):
        point_map = by_pairing.get(pairing, {})
        energies = {
            angle: point.energy_J_m2
            for angle, point in point_map.items()
            if point.usable and point.energy_J_m2 is not None
        }
        errors = {
            angle: point.error_J_m2
            for angle, point in point_map.items()
            if point.usable and point.error_J_m2 is not None
        }

        for angle_deg in target_angles:
            required = (
                angle_deg - 2 * step_deg,
                angle_deg - step_deg,
                angle_deg + step_deg,
                angle_deg + 2 * step_deg,
            )
            missing = [value for value in required if value not in energies]
            if missing:
                all_available = False
                rows.append(
                    {
                        "pairing": pairing,
                        "angle_deg": angle_deg,
                        "status": "missing_converged_energy",
                        "missing_angles_deg": " ".join(map(str, missing)),
                        "torque_per_area_N_m": "",
                        "energy_error_bound_N_m": "",
                        "stencil_sensitivity_N_m": "",
                        "combined_diagnostic_uncertainty_N_m": "",
                        "relative_energy_error_bound": "",
                        "relative_combined_diagnostic_uncertainty": "",
                    }
                )
                continue

            torque = five_point_torque(
                energies, angle_deg=angle_deg, step_deg=step_deg
            )
            torque_three = three_point_torque(
                energies, angle_deg=angle_deg, step_deg=step_deg
            )
            energy_bound = five_point_torque_error_bound(
                errors, angle_deg=angle_deg, step_deg=step_deg
            )
            stencil_sensitivity = abs(torque - torque_three)
            combined_diagnostic = energy_bound + stencil_sensitivity
            denominator = abs(torque)
            rows.append(
                {
                    "pairing": pairing,
                    "angle_deg": angle_deg,
                    "status": "computed",
                    "missing_angles_deg": "",
                    "torque_per_area_N_m": torque,
                    "energy_error_bound_N_m": energy_bound,
                    "stencil_sensitivity_N_m": stencil_sensitivity,
                    "combined_diagnostic_uncertainty_N_m": combined_diagnostic,
                    "relative_energy_error_bound": (
                        math.inf if denominator == 0.0 else energy_bound / denominator
                    ),
                    "relative_combined_diagnostic_uncertainty": (
                        math.inf
                        if denominator == 0.0
                        else combined_diagnostic / denominator
                    ),
                }
            )

    fields = (
        "pairing",
        "angle_deg",
        "status",
        "missing_angles_deg",
        "torque_per_area_N_m",
        "energy_error_bound_N_m",
        "stencil_sensitivity_N_m",
        "combined_diagnostic_uncertainty_N_m",
        "relative_energy_error_bound",
        "relative_combined_diagnostic_uncertainty",
    )
    _atomic_csv(torque_csv, fields, rows)

    temperatures = sorted({point.temperature_K for point in points})
    separations = sorted({point.separation_nm for point in points})
    metadata = {
        "profile": profile,
        "run_root": str(run_root),
        "temperature_K": None if not temperatures else temperatures[0],
        "separation_nm": None if not separations else separations[0],
        "step_deg": step_deg,
        "target_angles_deg": list(target_angles),
        "finite_difference": "five_point_centered",
        "stencil_sensitivity": "absolute_difference_between_five_and_three_point_centered_derivatives",
        "angle_derivative_unit": "radian",
        "torque_per_area_unit": "N/m",
        "energy_error_bound_is_formal_propagation": True,
        "stencil_sensitivity_is_formal_bound": False,
        "combined_diagnostic_uncertainty_is_formal_bound": False,
        "production_casimir_allowed": False,
        "energy_point_count": len(points),
        "usable_energy_point_count": sum(point.usable for point in points),
        "torque_row_count": len(rows),
        "computed_torque_row_count": sum(
            row["status"] == "computed" for row in rows
        ),
        "all_target_torques_available": all_available,
        "series_signatures": {
            pairing: sorted(
                {
                    point.series_signature
                    for point in points
                    if point.pairing == pairing
                }
            )
            for pairing in ("spm", "dwave")
        },
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_json.with_suffix(metadata_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_json)
    return energy_csv, torque_csv, metadata_json, all_available
