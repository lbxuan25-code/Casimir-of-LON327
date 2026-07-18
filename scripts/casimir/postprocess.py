from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
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
    status: str
    termination_reason: str
    matsubara_converged: bool
    energy_J_m2: float | None
    error_J_m2: float | None

    @property
    def usable(self) -> bool:
        return (
            self.matsubara_converged
            and self.energy_J_m2 is not None
            and self.error_J_m2 is not None
            and math.isfinite(self.energy_J_m2)
            and math.isfinite(self.error_J_m2)
            and self.error_J_m2 >= 0.0
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _case_regex(profile: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(spm|dwave)_T10K_d20nm_theta_([mp])(\d{{3}})deg_"
        rf"{re.escape(profile)}$"
    )


def decode_angle(sign: str, magnitude: str) -> int:
    value = int(magnitude)
    return -value if sign == "m" else value


def collect_energy_points(
    *,
    run_root: Path = DEFAULT_OUTPUT_ROOT,
    profile: str = PROFILE_NAME,
) -> list[EnergyPoint]:
    pattern = _case_regex(profile)
    output: list[EnergyPoint] = []

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
        summary = _read_json(run_dir / "summary.json")
        manifest = _read_json(run_dir / "manifest.json")
        payload = summary.get("pairings", {}).get(pairing, {})

        output.append(
            EnergyPoint(
                pairing=pairing,
                angle_deg=angle_deg,
                case=run_dir.name,
                status=str(summary.get("status", manifest.get("status", "missing"))),
                termination_reason=str(
                    summary.get(
                        "termination_reason",
                        manifest.get("termination_reason", ""),
                    )
                ),
                matsubara_converged=bool(summary.get("matsubara_converged", False)),
                energy_J_m2=_finite_float(
                    payload.get("finite_matsubara_partial_J_m2")
                ),
                error_J_m2=_finite_float(
                    payload.get("estimated_total_error_J_m2")
                ),
            )
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


def _write_energy_csv(path: Path, points: Sequence[EnergyPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "pairing",
        "angle_deg",
        "case",
        "status",
        "termination_reason",
        "matsubara_converged",
        "energy_J_m2",
        "energy_error_bound_J_m2",
        "usable_for_torque",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in sorted(points, key=lambda item: (item.pairing, item.angle_deg)):
            writer.writerow(
                {
                    "pairing": point.pairing,
                    "angle_deg": point.angle_deg,
                    "case": point.case,
                    "status": point.status,
                    "termination_reason": point.termination_reason,
                    "matsubara_converged": point.matsubara_converged,
                    "energy_J_m2": point.energy_J_m2,
                    "energy_error_bound_J_m2": point.error_J_m2,
                    "usable_for_torque": point.usable,
                }
            )


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
                        "torque_error_bound_N_m": "",
                        "relative_error_bound": "",
                    }
                )
                continue

            torque = five_point_torque(
                energies, angle_deg=angle_deg, step_deg=step_deg
            )
            bound = five_point_torque_error_bound(
                errors, angle_deg=angle_deg, step_deg=step_deg
            )
            relative = math.inf if torque == 0.0 else bound / abs(torque)
            rows.append(
                {
                    "pairing": pairing,
                    "angle_deg": angle_deg,
                    "status": "computed",
                    "missing_angles_deg": "",
                    "torque_per_area_N_m": torque,
                    "torque_error_bound_N_m": bound,
                    "relative_error_bound": relative,
                }
            )

    torque_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "pairing",
        "angle_deg",
        "status",
        "missing_angles_deg",
        "torque_per_area_N_m",
        "torque_error_bound_N_m",
        "relative_error_bound",
    )
    with torque_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "profile": profile,
        "run_root": str(run_root),
        "step_deg": step_deg,
        "target_angles_deg": list(target_angles),
        "finite_difference": "five_point_centered",
        "angle_derivative_unit": "radian",
        "torque_per_area_unit": "N/m",
        "energy_point_count": len(points),
        "usable_energy_point_count": sum(point.usable for point in points),
        "torque_row_count": len(rows),
        "computed_torque_row_count": sum(
            row["status"] == "computed" for row in rows
        ),
        "all_target_torques_available": all_available,
    }
    metadata_json.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return energy_csv, torque_csv, metadata_json, all_available
