"""Incremental certified microscopic-point providers for adaptive integration.

The providers are thin orchestration layers over the existing production transverse
point certifier.  They never evaluate microscopic response themselves.  Exact model-q
coordinates are keyed by their IEEE-754 hexadecimal representations, so repeated
adaptive refinement rounds reuse only bitwise-identical points.

``CertifiedOuterQProvider`` preserves the original fixed-frequency cache contract.
``FrequencyExtendableCertifiedOuterQProvider`` additionally permits a cumulative
Matsubara set to grow under one unchanged microscopic policy.  Only newly requested
Matsubara indices are sent back to the production certifier.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    _CertificationRun,
    _run_transverse_certifier,
)
from .fixed_outer_q import OuterQNodeManifest

_CACHE_SCHEMA_V1 = "certified-outer-q-point-cache-v1"
_CACHE_SCHEMA_V2 = "certified-outer-q-point-cache-v2-matsubara-extendable"


class CertifiedPointCacheError(RuntimeError):
    """Raised when a persisted point cache is incompatible or malformed."""


@dataclass(frozen=True)
class CertifiedPointBatch:
    """Result of one incremental provider request."""

    point_results: tuple[Mapping[str, Any], ...]
    unresolved_points: tuple[Mapping[str, Any], ...]
    requested_q_count: int
    new_q_count: int
    cache_hit_q_count: int
    certification_batches: int
    requested_point_count: int = 0
    new_point_count: int = 0
    cache_hit_point_count: int = 0

    @property
    def all_established(self) -> bool:
        return not self.unresolved_points


Runner = Callable[[FixedCasimirConfig, OuterQNodeManifest, Path], _CertificationRun]


def _q_key(q: Sequence[float]) -> tuple[str, str]:
    array = np.asarray(q, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError("each model-q point must contain two finite values")
    return (float(array[0]).hex(), float(array[1]).hex())


def _entry_key(pairing: str, n: int, q_key: tuple[str, str]) -> str:
    return "|".join((str(pairing), str(int(n)), q_key[0], q_key[1]))


def _stable_q_label(q_key: tuple[str, str]) -> str:
    digest = hashlib.sha256("|".join(q_key).encode("ascii")).hexdigest()[:20]
    return f"adaptive_q_{digest}"


def _point_policy_payload(
    config: FixedCasimirConfig,
    *,
    frequency_extendable: bool = False,
) -> dict[str, Any]:
    """Return only inputs that can change one certified microscopic point.

    For the v2 cache the requested Matsubara *set* is orchestration state rather than
    microscopic policy.  The Matsubara index remains part of every cache-entry key.
    Temperature and all other frequency-defining physical inputs remain fingerprinted.
    """

    payload = config.as_dict()
    for name in (
        "u_max_values",
        "radial_orders",
        "angular_orders",
        "angular_offsets",
        "outer_rtol",
        "outer_atol_J_m2",
        "transverse_checkpoint_path",
    ):
        payload.pop(name, None)
    if frequency_extendable:
        payload.pop("matsubara_indices", None)
    return payload


def _policy_fingerprint(
    config: FixedCasimirConfig,
    *,
    frequency_extendable: bool = False,
) -> str:
    encoded = json.dumps(
        _point_policy_payload(
            config,
            frequency_extendable=frequency_extendable,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def certified_primary_logdet(point: Mapping[str, Any]) -> float:
    """Extract the finite hard-physical primary audit-shift logdet."""

    sweet = point.get("sweet_spot", {})
    if sweet.get("status") != "established":
        raise ValueError("point sweet spot is not established")
    audit_n = int(sweet["audit_N"])
    row = next(
        (
            item
            for item in point.get("history", [])
            if int(item.get("N", -1)) == audit_n
        ),
        None,
    )
    if row is None:
        raise ValueError("point history does not contain its audit_N")
    shifts = row.get("shifts", {})
    if not shifts:
        raise ValueError("audit_N history row has no shift states")
    primary = next(iter(shifts.values()))
    value = float(primary["two_plate_logdet"])
    if not np.isfinite(value) or not bool(primary.get("hard_physical_passed")):
        raise ValueError("primary audit shift is not a finite hard-physical point")
    cross_shift = row.get("two_plate_logdet_cross_shift", {})
    if cross_shift and not bool(cross_shift.get("passed")):
        raise ValueError("point audit cross-shift comparison did not pass")
    return value


class CertifiedOuterQProvider:
    """Incrementally certify and cache exact outer-Q microscopic points."""

    def __init__(
        self,
        config: FixedCasimirConfig,
        *,
        cache_path: Path | None = None,
        runner: Runner | None = None,
        certifier_q_batch_size: int = 256,
        _frequency_extendable: bool = False,
    ) -> None:
        if not isinstance(config, FixedCasimirConfig):
            raise TypeError("config must be a FixedCasimirConfig")
        self.config = config
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._runner = _run_transverse_certifier if runner is None else runner
        batch_size = int(certifier_q_batch_size)
        if batch_size <= 0:
            raise ValueError("certifier_q_batch_size must be positive")
        self.certifier_q_batch_size = batch_size
        self._frequency_extendable = bool(_frequency_extendable)
        self._cache_schema = (
            _CACHE_SCHEMA_V2 if self._frequency_extendable else _CACHE_SCHEMA_V1
        )
        self._policy_fingerprint = _policy_fingerprint(
            config,
            frequency_extendable=self._frequency_extendable,
        )
        self._entries: dict[str, Mapping[str, Any]] = {}
        self._q_by_entry: dict[str, tuple[str, str]] = {}
        self.certification_batches = 0
        self.certification_failed_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0
        self.certifier_wall_seconds = 0.0
        self.certifier_reported_level_wall_seconds = 0.0
        self.certifier_material_build_seconds = 0.0
        self.certifier_context_wall_seconds = 0.0
        self.cache_load_seconds = 0.0
        self.cache_save_seconds = 0.0
        self.cache_save_count = 0
        self.cache_file_bytes = 0
        self.certifier_batch_records: list[dict[str, Any]] = []
        if self.cache_path is not None and self.cache_path.exists():
            started = perf_counter()
            try:
                self._load()
            finally:
                self.cache_load_seconds += float(perf_counter() - started)
            try:
                self.cache_file_bytes = int(self.cache_path.stat().st_size)
            except OSError:
                self.cache_file_bytes = 0

    @property
    def cached_point_count(self) -> int:
        return len(self._entries)

    @property
    def unique_q_count(self) -> int:
        return len(set(self._q_by_entry.values()))

    @property
    def frequency_extendable(self) -> bool:
        return self._frequency_extendable

    def performance_statistics(self) -> dict[str, Any]:
        """Return orchestration telemetry without changing numerical policy."""

        return {
            "cached_point_count": int(self.cached_point_count),
            "unique_q_count": int(self.unique_q_count),
            "certification_batches": int(self.certification_batches),
            "certification_failed_batches": int(
                self.certification_failed_batches
            ),
            "certification_attempts": int(
                self.certification_batches + self.certification_failed_batches
            ),
            "certifier_q_batch_size": int(self.certifier_q_batch_size),
            "requested_q_evaluations": int(self.requested_q_evaluations),
            "new_q_evaluations": int(self.new_q_evaluations),
            "cache_hit_q_evaluations": int(self.cache_hit_q_evaluations),
            "requested_point_evaluations": int(self.requested_point_evaluations),
            "new_point_evaluations": int(self.new_point_evaluations),
            "cache_hit_point_evaluations": int(
                self.cache_hit_point_evaluations
            ),
            "certifier_wall_seconds": float(self.certifier_wall_seconds),
            "certifier_reported_level_wall_seconds": float(
                self.certifier_reported_level_wall_seconds
            ),
            "certifier_material_build_seconds": float(
                self.certifier_material_build_seconds
            ),
            "certifier_context_wall_seconds": float(
                self.certifier_context_wall_seconds
            ),
            "cache_load_seconds": float(self.cache_load_seconds),
            "cache_save_seconds": float(self.cache_save_seconds),
            "cache_save_count": int(self.cache_save_count),
            "cache_file_bytes": int(self.cache_file_bytes),
            "certifier_batch_records": [
                dict(record) for record in self.certifier_batch_records
            ],
        }

    def _consume_certifier_telemetry(
        self,
        payload: Mapping[str, Any],
        *,
        wall_seconds: float,
        requested_q_count: int,
        requested_point_count: int,
        matsubara_indices: Sequence[int],
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        level_wall = 0.0
        material_build = 0.0
        context_wall = 0.0
        levels = payload.get("execution_levels", ())
        if isinstance(levels, Sequence):
            for level in levels:
                if not isinstance(level, Mapping):
                    continue
                level_wall += float(level.get("level_wall_seconds", 0.0))
                pairings = level.get("pairings", {})
                if not isinstance(pairings, Mapping):
                    continue
                for records in pairings.values():
                    if not isinstance(records, Sequence):
                        continue
                    for record in records:
                        if not isinstance(record, Mapping):
                            continue
                        material_build += float(
                            record.get("material_build_seconds", 0.0)
                        )
                        context_wall += float(
                            record.get("context_wall_seconds", 0.0)
                        )
        self.certifier_reported_level_wall_seconds += level_wall
        self.certifier_material_build_seconds += material_build
        self.certifier_context_wall_seconds += context_wall
        self.certifier_batch_records.append(
            {
                "batch_index": len(self.certifier_batch_records),
                "status": "succeeded",
                "requested_q_count": int(requested_q_count),
                "requested_point_count": int(requested_point_count),
                "matsubara_indices": [
                    int(value) for value in matsubara_indices
                ],
                "certifier_wall_seconds": float(wall_seconds),
                "reported_level_wall_seconds": float(level_wall),
                "material_build_seconds": float(material_build),
                "context_wall_seconds": float(context_wall),
                "stdout_tail": str(stdout)[-4000:],
                "stderr_tail": str(stderr)[-4000:],
            }
        )

    def _record_certifier_failure(
        self,
        *,
        wall_seconds: float,
        requested_q_count: int,
        requested_point_count: int,
        matsubara_indices: Sequence[int],
        exception: Exception,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.certifier_batch_records.append(
            {
                "batch_index": len(self.certifier_batch_records),
                "status": "failed",
                "requested_q_count": int(requested_q_count),
                "requested_point_count": int(requested_point_count),
                "matsubara_indices": [
                    int(value) for value in matsubara_indices
                ],
                "certifier_wall_seconds": float(wall_seconds),
                "reported_level_wall_seconds": 0.0,
                "material_build_seconds": 0.0,
                "context_wall_seconds": 0.0,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stdout_tail": str(stdout)[-4000:],
                "stderr_tail": str(stderr)[-4000:],
            }
        )

    def reconfigure(self, config: FixedCasimirConfig) -> None:
        """Switch the active cumulative Matsubara request under the same policy.

        The original v1 provider is deliberately immutable.  The v2 provider accepts
        only configurations whose frequency-independent point-policy fingerprint is
        identical.  Pairings, temperature, shifts, N ladders and every physical gate
        therefore remain frozen while the non-negative Matsubara set may grow.
        """

        if not self._frequency_extendable:
            raise TypeError("this certified point provider has a fixed Matsubara set")
        if not isinstance(config, FixedCasimirConfig):
            raise TypeError("config must be a FixedCasimirConfig")
        fingerprint = _policy_fingerprint(config, frequency_extendable=True)
        if fingerprint != self._policy_fingerprint:
            raise CertifiedPointCacheError(
                "Matsubara extension changes the microscopic point-policy fingerprint"
            )
        self.config = config

    def count_new_q(self, q_model: np.ndarray) -> int:
        points = self._unique_points(q_model)
        return sum(bool(self._missing_indices(q_key)) for q_key, _ in points)

    def point_result(
        self,
        pairing: str,
        n: int,
        q: Sequence[float],
    ) -> Mapping[str, Any] | None:
        key = _entry_key(str(pairing), int(n), _q_key(q))
        return self._entries.get(key)

    def primary_logdet(self, pairing: str, n: int, q: Sequence[float]) -> float:
        point = self.point_result(pairing, n, q)
        if point is None:
            raise KeyError("the requested microscopic point is not cached")
        return certified_primary_logdet(point)

    def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch:
        points = self._unique_points(q_model)
        pairings = tuple(self.config.pairings)
        indices = tuple(self.config.matsubara_indices)
        requested_point_count = len(points) * len(pairings) * len(indices)
        self.requested_q_evaluations += len(points)
        self.requested_point_evaluations += requested_point_count

        missing_by_q = {
            q_key: self._missing_indices(q_key) for q_key, _ in points
        }
        incomplete = {q_key for q_key, missing in missing_by_q.items() if missing}
        cache_hit_q_count = len(points) - len(incomplete)
        self.cache_hit_q_evaluations += cache_hit_q_count

        if self._frequency_extendable:
            groups: dict[
                tuple[int, ...],
                list[tuple[tuple[str, str], tuple[float, float]]],
            ] = {}
            for q_key, q in points:
                missing = missing_by_q[q_key]
                if missing:
                    groups.setdefault(missing, []).append((q_key, q))
        else:
            missing_points = [
                (q_key, q) for q_key, q in points if q_key in incomplete
            ]
            groups = {indices: missing_points} if missing_points else {}

        new_point_count = 0
        new_batch_count = 0
        for missing_indices, group in groups.items():
            if not group:
                continue
            run_config = (
                replace(self.config, matsubara_indices=missing_indices)
                if self._frequency_extendable
                else self.config
            )
            total_chunks = (
                len(group) + self.certifier_q_batch_size - 1
            ) // self.certifier_q_batch_size
            for chunk_index, start in enumerate(
                range(0, len(group), self.certifier_q_batch_size),
                start=1,
            ):
                chunk = group[start : start + self.certifier_q_batch_size]
                labels = tuple(
                    _stable_q_label(q_key) for q_key, _ in chunk
                )
                q_values = np.asarray([q for _, q in chunk], dtype=float)
                manifest = OuterQNodeManifest(
                    labels=labels,
                    q_model=q_values,
                    grids={},
                    labels_by_spec={},
                )
                chunk_point_count = (
                    len(chunk)
                    * len(run_config.pairings)
                    * len(run_config.matsubara_indices)
                )
                started = perf_counter()
                certification: _CertificationRun | None = None
                try:
                    with TemporaryDirectory(
                        prefix="lno327-adaptive-point-batch-"
                    ) as temporary:
                        output = Path(temporary) / "certification.json"
                        certification = self._runner(
                            run_config, manifest, output
                        )
                    wall_seconds = float(perf_counter() - started)
                    self.certifier_wall_seconds += wall_seconds
                    self._consume_payload(
                        certification.payload,
                        labels=labels,
                        q_keys=tuple(q_key for q_key, _ in chunk),
                        requested_config=run_config,
                    )
                    self._consume_certifier_telemetry(
                        certification.payload,
                        wall_seconds=wall_seconds,
                        requested_q_count=len(chunk),
                        requested_point_count=chunk_point_count,
                        matsubara_indices=run_config.matsubara_indices,
                        stdout=certification.stdout,
                        stderr=certification.stderr,
                    )
                except Exception as exc:
                    wall_seconds = float(perf_counter() - started)
                    self.certifier_wall_seconds += wall_seconds
                    self.certification_failed_batches += 1
                    self._record_certifier_failure(
                        wall_seconds=wall_seconds,
                        requested_q_count=len(chunk),
                        requested_point_count=chunk_point_count,
                        matsubara_indices=run_config.matsubara_indices,
                        exception=exc,
                        stdout=(
                            "" if certification is None else certification.stdout
                        ),
                        stderr=(
                            "" if certification is None else certification.stderr
                        ),
                    )
                    if self.cache_path is not None:
                        self._save()
                    raise FixedCasimirExecutionError(
                        "certifier q batch failed "
                        f"(group_q={len(group)}, chunk={chunk_index}/{total_chunks}, "
                        f"chunk_q={len(chunk)}, matsubara_indices="
                        f"{tuple(run_config.matsubara_indices)}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                self.certification_batches += 1
                self.new_q_evaluations += len(chunk)
                self.new_point_evaluations += chunk_point_count
                new_point_count += chunk_point_count
                new_batch_count += 1

                # Large refinement requests are checkpointed after every successful
                # chunk so a later failed chunk cannot discard already certified q.
                if (
                    self.cache_path is not None
                    and len(group) > self.certifier_q_batch_size
                ):
                    self._save()

        cache_hit_point_count = requested_point_count - new_point_count
        self.cache_hit_point_evaluations += cache_hit_point_count
        if groups:
            self._save()

        rows: list[Mapping[str, Any]] = []
        unresolved: list[Mapping[str, Any]] = []
        for q_key, _ in points:
            label = _stable_q_label(q_key)
            for pairing in pairings:
                for n in indices:
                    key = _entry_key(pairing, n, q_key)
                    point = self._entries.get(key)
                    if point is None:
                        unresolved.append(
                            {
                                "pairing": pairing,
                                "n": int(n),
                                "q_label": label,
                                "q_hex": list(q_key),
                                "reason": "missing_certification_result",
                            }
                        )
                        continue
                    rows.append(point)
                    try:
                        certified_primary_logdet(point)
                    except (KeyError, TypeError, ValueError) as exc:
                        unresolved.append(
                            {
                                "pairing": pairing,
                                "n": int(n),
                                "q_label": label,
                                "q_hex": list(q_key),
                                "reason": str(exc),
                            }
                        )

        return CertifiedPointBatch(
            point_results=tuple(rows),
            unresolved_points=tuple(unresolved),
            requested_q_count=len(points),
            new_q_count=len(incomplete),
            cache_hit_q_count=cache_hit_q_count,
            certification_batches=new_batch_count,
            requested_point_count=requested_point_count,
            new_point_count=new_point_count,
            cache_hit_point_count=cache_hit_point_count,
        )

    def _unique_points(
        self,
        q_model: np.ndarray,
    ) -> tuple[tuple[tuple[str, str], tuple[float, float]], ...]:
        array = np.asarray(q_model, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError("q_model must be a finite array with shape (N, 2)")
        unique: dict[tuple[str, str], tuple[float, float]] = {}
        for q in array:
            key = _q_key(q)
            unique.setdefault(key, (float(q[0]), float(q[1])))
        return tuple(unique.items())

    def _missing_indices(self, q_key: tuple[str, str]) -> tuple[int, ...]:
        return tuple(
            int(n)
            for n in self.config.matsubara_indices
            if any(
                _entry_key(pairing, n, q_key) not in self._entries
                for pairing in self.config.pairings
            )
        )

    def _consume_payload(
        self,
        payload: Mapping[str, Any],
        *,
        labels: tuple[str, ...],
        q_keys: tuple[tuple[str, str], ...],
        requested_config: FixedCasimirConfig,
    ) -> None:
        if payload.get("schema") != "transverse-point-sweet-spot-v4":
            raise CertifiedPointCacheError(
                "transverse certifier returned an unexpected schema"
            )
        q_by_label = dict(zip(labels, q_keys, strict=True))
        for point in payload.get("point_results", []):
            label = str(point.get("q_label", ""))
            q_key = q_by_label.get(label)
            if q_key is None:
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unknown q label"
                )
            pairing = str(point.get("pairing", ""))
            n = int(point.get("n", -1))
            if (
                pairing not in requested_config.pairings
                or n not in requested_config.matsubara_indices
            ):
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unrequested point"
                )
            key = _entry_key(pairing, n, q_key)
            self._entries[key] = dict(point)
            self._q_by_entry[key] = q_key

    def _load(self) -> None:
        assert self.cache_path is not None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CertifiedPointCacheError(f"cannot read point cache: {exc}") from exc
        if payload.get("schema") != self._cache_schema:
            raise CertifiedPointCacheError("point cache schema mismatch")
        if payload.get("policy_fingerprint") != self._policy_fingerprint:
            raise CertifiedPointCacheError("point cache policy fingerprint mismatch")
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise CertifiedPointCacheError("point cache entries must be a list")
        for entry in entries:
            try:
                pairing = str(entry["pairing"])
                n = int(entry["n"])
                q_key = (str(entry["qx_hex"]), str(entry["qy_hex"]))
                point = dict(entry["point_result"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CertifiedPointCacheError("malformed point cache entry") from exc
            if n < 0:
                raise CertifiedPointCacheError("point cache contains a negative Matsubara index")
            key = _entry_key(pairing, n, q_key)
            self._entries[key] = point
            self._q_by_entry[key] = q_key

    def _save(self) -> None:
        if self.cache_path is None:
            return
        started = perf_counter()
        entries = []
        for key in sorted(self._entries):
            pairing, n_token, qx_hex, qy_hex = key.split("|", 3)
            entries.append(
                {
                    "pairing": pairing,
                    "n": int(n_token),
                    "qx_hex": qx_hex,
                    "qy_hex": qy_hex,
                    "point_result": dict(self._entries[key]),
                }
            )
        payload = {
            "schema": self._cache_schema,
            "policy_fingerprint": self._policy_fingerprint,
            "frequency_extendable": self._frequency_extendable,
            "active_matsubara_indices": list(self.config.matsubara_indices),
            "point_policy": _point_policy_payload(
                self.config,
                frequency_extendable=self._frequency_extendable,
            ),
            "entries": entries,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)
        self.cache_save_seconds += float(perf_counter() - started)
        self.cache_save_count += 1
        try:
            self.cache_file_bytes = int(self.cache_path.stat().st_size)
        except OSError:
            self.cache_file_bytes = 0

        telemetry_path = self.cache_path.with_suffix(".telemetry.json")
        telemetry_temporary = telemetry_path.with_suffix(
            telemetry_path.suffix + ".tmp"
        )
        telemetry_temporary.write_text(
            json.dumps(
                self.performance_statistics(),
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        telemetry_temporary.replace(telemetry_path)


class FrequencyExtendableCertifiedOuterQProvider(CertifiedOuterQProvider):
    """Certified point provider whose cumulative Matsubara set may grow safely."""

    def __init__(
        self,
        config: FixedCasimirConfig,
        *,
        cache_path: Path | None = None,
        runner: Runner | None = None,
        certifier_q_batch_size: int = 256,
    ) -> None:
        super().__init__(
            config,
            cache_path=cache_path,
            runner=runner,
            certifier_q_batch_size=certifier_q_batch_size,
            _frequency_extendable=True,
        )


__all__ = [
    "CertifiedOuterQProvider",
    "CertifiedPointBatch",
    "CertifiedPointCacheError",
    "FrequencyExtendableCertifiedOuterQProvider",
    "certified_primary_logdet",
]
