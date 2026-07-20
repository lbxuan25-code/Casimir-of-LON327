from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig

from .cache_migration import (
    CACHE_SCHEMA,
    _atomic_json,
    _point_config_from_run_config,
    _read_json_mapping,
    _validated_entries,
)
from .config import case_name


@dataclass(frozen=True)
class ExtensionPreparationReport:
    pairing: str
    source_cache: Path
    target_cache: Path
    mode: str
    source_N_candidates: tuple[int, ...]
    target_N_candidates: tuple[int, ...]
    source_entry_count: int
    retained_entry_count: int
    dropped_unresolved_count: int
    dropped_identities: tuple[tuple[str, int, str, str], ...]
    source_cache_sha256: str
    target_cache_sha256: str
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pilot-cache-extension-report-v1",
            "pairing": self.pairing,
            "source_cache": str(self.source_cache),
            "target_cache": str(self.target_cache),
            "mode": self.mode,
            "source_N_candidates": list(self.source_N_candidates),
            "target_N_candidates": list(self.target_N_candidates),
            "source_entry_count": self.source_entry_count,
            "retained_entry_count": self.retained_entry_count,
            "dropped_unresolved_count": self.dropped_unresolved_count,
            "dropped_identities": [list(value) for value in self.dropped_identities],
            "source_cache_sha256": self.source_cache_sha256,
            "target_cache_sha256": self.target_cache_sha256,
            "skipped": self.skipped,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash cache {path}: {exc}") from exc
    return digest.hexdigest()


def _entry_identity(entry: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(entry["pairing"]),
        int(entry["n"]),
        str(entry["qx_hex"]),
        str(entry["qy_hex"]),
    )


def _extension_mode(
    source: FixedCasimirConfig,
    target: FixedCasimirConfig,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    left = certified_point_policy_payload(source, frequency_extendable=True)
    right = certified_point_policy_payload(target, frequency_extendable=True)
    source_N = tuple(int(value) for value in left.pop("N_candidates"))
    target_N = tuple(int(value) for value in right.pop("N_candidates"))
    if left != right:
        raise ValueError(
            "pilot cache extension requires an unchanged microscopic point policy "
            "apart from a strict N_candidates prefix extension"
        )
    if target_N == source_N:
        return "identical_point_policy", source_N, target_N
    if (
        len(target_N) <= len(source_N)
        or target_N[: len(source_N)] != source_N
    ):
        raise ValueError(
            "target N_candidates must equal the source ladder or strictly extend it "
            "with the complete source ladder as a prefix"
        )
    return "N_ladder_prefix_extension", source_N, target_N


def prepare_extension_cache(
    *,
    pairing: str,
    source_run_dir: Path,
    target_run_dir: Path,
    target_point_config: FixedCasimirConfig,
) -> ExtensionPreparationReport:
    source_run = Path(source_run_dir)
    target_run = Path(target_run_dir)
    if source_run.resolve() == target_run.resolve():
        raise ValueError("source and target pilot run directories must be different")

    source_cache = source_run / "cache" / "certified_points.json"
    target_cache = target_run / "cache" / "certified_points.json"
    if not source_cache.is_file():
        raise FileNotFoundError(f"source pilot cache does not exist: {source_cache}")

    source_run_config = _read_json_mapping(
        source_run / "config.json",
        label="source run config",
    )
    source_config = _point_config_from_run_config(source_run_config)
    mode, source_N, target_N = _extension_mode(source_config, target_point_config)

    target_fingerprint = certified_point_policy_fingerprint(
        target_point_config,
        frequency_extendable=True,
    )
    target_policy = certified_point_policy_payload(
        target_point_config,
        frequency_extendable=True,
    )
    source_sha256 = _sha256(source_cache)
    source_fingerprint = certified_point_policy_fingerprint(
        source_config,
        frequency_extendable=True,
    )
    source_policy = certified_point_policy_payload(
        source_config,
        frequency_extendable=True,
    )
    source_payload = _read_json_mapping(source_cache, label="source cache")
    entries = _validated_entries(
        source_payload,
        path=source_cache,
        expected_fingerprint=source_fingerprint,
        expected_policy=source_policy,
        allow_legacy_scheduling_fingerprint=True,
    )
    if any(str(entry["pairing"]) != pairing for entry in entries):
        raise ValueError(
            f"source cache contains entries outside requested pairing {pairing!r}"
        )

    source_unresolved = tuple(
        _entry_identity(entry)
        for entry in entries
        if (
            entry["point_result"]
            .get("sweet_spot", {})
            .get("status")
            != "established"
        )
    )
    if target_cache.exists():
        target_payload = _read_json_mapping(target_cache, label="target cache")
        target_entries = _validated_entries(
            target_payload,
            path=target_cache,
            expected_fingerprint=target_fingerprint,
            expected_policy=target_policy,
        )
        return ExtensionPreparationReport(
            pairing=pairing,
            source_cache=source_cache,
            target_cache=target_cache,
            mode="existing_target_cache",
            source_N_candidates=source_N,
            target_N_candidates=target_N,
            source_entry_count=len(entries),
            retained_entry_count=len(target_entries),
            dropped_unresolved_count=len(source_unresolved),
            dropped_identities=source_unresolved,
            source_cache_sha256=source_sha256,
            target_cache_sha256=_sha256(target_cache),
            skipped=True,
        )

    retained: list[dict[str, Any]] = []
    dropped: list[tuple[str, int, str, str]] = []
    for entry in entries:
        status = (
            entry["point_result"]
            .get("sweet_spot", {})
            .get("status")
        )
        if status == "established":
            retained.append(entry)
        else:
            dropped.append(_entry_identity(entry))

    retained_indices = sorted({int(entry["n"]) for entry in retained})
    target_payload = {
        "schema": CACHE_SCHEMA,
        "policy_fingerprint": target_fingerprint,
        "frequency_extendable": True,
        "active_matsubara_indices": (
            retained_indices
            if retained_indices
            else list(target_point_config.matsubara_indices)
        ),
        "point_policy": target_policy,
        "entries": retained,
    }
    _atomic_json(target_cache, target_payload, compact=True)
    report = ExtensionPreparationReport(
        pairing=pairing,
        source_cache=source_cache,
        target_cache=target_cache,
        mode=mode,
        source_N_candidates=source_N,
        target_N_candidates=target_N,
        source_entry_count=len(entries),
        retained_entry_count=len(retained),
        dropped_unresolved_count=len(dropped),
        dropped_identities=tuple(dropped),
        source_cache_sha256=source_sha256,
        target_cache_sha256=_sha256(target_cache),
    )
    _atomic_json(target_cache.with_name("extension_report.json"), report.as_dict())
    return report


def prepare_pilot_extension_caches(
    *,
    pairings: tuple[str, ...],
    output_root: Path,
    source_profile: str,
    target_profile: str,
    target_configs: Mapping[str, FixedCasimirConfig],
) -> tuple[ExtensionPreparationReport, ...]:
    if source_profile == target_profile:
        raise ValueError("source and target pilot profiles must be different")
    reports = []
    for pairing in pairings:
        reports.append(
            prepare_extension_cache(
                pairing=pairing,
                source_run_dir=output_root
                / case_name(pairing, 0, profile=source_profile),
                target_run_dir=output_root
                / case_name(pairing, 0, profile=target_profile),
                target_point_config=target_configs[pairing],
            )
        )
    return tuple(reports)
