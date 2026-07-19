from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.casimir.fixed_transverse_point_certification import (
    ENVELOPE_LEVELS,
    assess_frequency_level,
    assess_oscillatory_envelope,
)

from .config import case_name

CACHE_SCHEMA = "certified-outer-q-point-cache-v2-matsubara-extendable"


@dataclass(frozen=True)
class MigrationReport:
    pairing: str
    source_cache: Path
    target_cache: Path
    source_entry_count: int
    target_entry_count: int
    established_before: int
    established_after: int
    newly_established: int
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairing": self.pairing,
            "source_cache": str(self.source_cache),
            "target_cache": str(self.target_cache),
            "source_entry_count": self.source_entry_count,
            "target_entry_count": self.target_entry_count,
            "established_before": self.established_before,
            "established_after": self.established_after,
            "newly_established": self.newly_established,
            "skipped": self.skipped,
        }


def _atomic_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _point_config_from_run_config(payload: Mapping[str, Any]) -> FixedCasimirConfig:
    try:
        point = payload["outer_tail_config"]["joint_config"]["radial_config"][
            "point_config"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("run config does not contain a microscopic point config") from exc
    return FixedCasimirConfig(**dict(point))


def _assert_relaxation_only(
    source: FixedCasimirConfig,
    target: FixedCasimirConfig,
) -> None:
    left = certified_point_policy_payload(source, frequency_extendable=True)
    right = certified_point_policy_payload(target, frequency_extendable=True)
    source_rtol = float(left.pop("logdet_rtol"))
    target_rtol = float(right.pop("logdet_rtol"))
    source_atol = float(left.get("logdet_atol"))
    target_atol = float(right.get("logdet_atol"))
    if left != right:
        raise ValueError(
            "cache migration is allowed only when physical and numerical point policy "
            "is unchanged apart from logdet_rtol"
        )
    if target_rtol < source_rtol or target_atol != source_atol:
        raise ValueError("target logdet policy is not a pure relative-tolerance relaxation")


def reassess_point(
    point: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    output = dict(point)
    rebuilt: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] | None = None
    consecutive = 0
    established: dict[str, Any] | None = None
    for raw_row in point.get("history", []):
        if not isinstance(raw_row, Mapping):
            raise ValueError("point history contains a malformed row")
        row = dict(raw_row)
        shifts = row.get("shifts")
        if not isinstance(shifts, Mapping) or not shifts:
            raise ValueError("point history row has no shift states")
        current = {str(key): dict(value) for key, value in shifts.items()}
        assessment = assess_frequency_level(
            current_by_shift=current,
            previous_by_shift=previous,
            rtol=rtol,
            atol=atol,
        )
        consecutive = consecutive + 1 if assessment["accepted_transition"] else 0
        row.update(assessment)
        row["consecutive_accepted_transitions"] = consecutive
        rebuilt.append(row)
        envelope = assess_oscillatory_envelope(
            rebuilt,
            rtol=rtol,
            atol=atol,
        )
        row["oscillatory_envelope"] = envelope
        strict_ready = consecutive >= required_consecutive_passes
        envelope_ready = bool(envelope["passed"])
        if strict_ready or envelope_ready:
            if len(rebuilt) < 2:
                raise ValueError("reassessed convergence has no previous N level")
            established = {
                "status": "established",
                "establishment_mode": (
                    "strict_consecutive_adjacent"
                    if strict_ready
                    else "three_level_oscillatory_envelope"
                ),
                "working_N": int(rebuilt[-2]["N"]),
                "audit_N": int(rebuilt[-1]["N"]),
                "required_consecutive_passes": required_consecutive_passes,
                "envelope_levels": ENVELOPE_LEVELS,
                "envelope_N_window": list(envelope["N_window"]),
                "criterion": (
                    "universal hard physical closure and cross-shift stability plus "
                    "either consecutive adjacent-N convergence or a three-level "
                    "absolute-first, relative-fallback oscillatory envelope"
                ),
            }
            break
        previous = current
    output["history"] = rebuilt
    output["sweet_spot"] = established or {
        "status": "not_established",
        "establishment_mode": None,
        "working_N": None,
        "audit_N": None,
    }
    return output


def migrate_cache(
    *,
    pairing: str,
    source_run_dir: Path,
    target_run_dir: Path,
    target_point_config: FixedCasimirConfig,
) -> MigrationReport:
    source_cache = source_run_dir / "cache" / "certified_points.json"
    target_cache = target_run_dir / "cache" / "certified_points.json"
    if target_cache.exists():
        payload = json.loads(target_cache.read_text(encoding="utf-8"))
        count = len(payload.get("entries", []))
        established = sum(
            entry.get("point_result", {}).get("sweet_spot", {}).get("status")
            == "established"
            for entry in payload.get("entries", [])
        )
        return MigrationReport(
            pairing, source_cache, target_cache, count, count, established, established, 0, True
        )
    if not source_cache.exists():
        return MigrationReport(
            pairing, source_cache, target_cache, 0, 0, 0, 0, 0, True
        )
    source_config_path = source_run_dir / "config.json"
    source_config = _point_config_from_run_config(
        json.loads(source_config_path.read_text(encoding="utf-8"))
    )
    _assert_relaxation_only(source_config, target_point_config)
    payload = json.loads(source_cache.read_text(encoding="utf-8"))
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("source pilot cache has an incompatible schema")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source pilot cache entries must be a list")
    before = 0
    after = 0
    migrated_entries: list[dict[str, Any]] = []
    for entry in entries:
        point = dict(entry["point_result"])
        was_established = point.get("sweet_spot", {}).get("status") == "established"
        before += int(was_established)
        reassessed = reassess_point(
            point,
            rtol=target_point_config.logdet_rtol,
            atol=target_point_config.logdet_atol,
            required_consecutive_passes=target_point_config.required_consecutive_passes,
        )
        is_established = reassessed["sweet_spot"]["status"] == "established"
        after += int(is_established)
        migrated_entries.append({**dict(entry), "point_result": reassessed})
    target_payload = {
        "schema": CACHE_SCHEMA,
        "policy_fingerprint": certified_point_policy_fingerprint(
            target_point_config, frequency_extendable=True
        ),
        "frequency_extendable": True,
        "active_matsubara_indices": list(target_point_config.matsubara_indices),
        "point_policy": certified_point_policy_payload(
            target_point_config, frequency_extendable=True
        ),
        "entries": migrated_entries,
    }
    _atomic_json(target_cache, target_payload, compact=True)
    report = MigrationReport(
        pairing=pairing,
        source_cache=source_cache,
        target_cache=target_cache,
        source_entry_count=len(entries),
        target_entry_count=len(migrated_entries),
        established_before=before,
        established_after=after,
        newly_established=after - before,
    )
    _atomic_json(target_cache.with_name("migration_report.json"), report.as_dict())
    return report


def migrate_pilot_caches(
    *,
    pairings: tuple[str, ...],
    output_root: Path,
    source_profile: str,
    target_profile: str,
    target_configs: Mapping[str, FixedCasimirConfig],
) -> tuple[MigrationReport, ...]:
    reports = []
    for pairing in pairings:
        reports.append(migrate_cache(
            pairing=pairing,
            source_run_dir=output_root / case_name(pairing, 0, profile=source_profile),
            target_run_dir=output_root / case_name(pairing, 0, profile=target_profile),
            target_point_config=target_configs[pairing],
        ))
    return tuple(reports)
