"""Incremental certified microscopic-point provider for outer-Q integration.

The provider is a thin orchestration layer over the existing production transverse
point certifier.  It never evaluates microscopic response itself.  Exact model-q
coordinates are keyed by their IEEE-754 hexadecimal representations, so repeated
adaptive refinement rounds reuse only bitwise-identical points.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .fixed_chain import (
    FixedCasimirConfig,
    _CertificationRun,
    _run_transverse_certifier,
)
from .fixed_outer_q import OuterQNodeManifest

_CACHE_SCHEMA = "certified-outer-q-point-cache-v1"


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


def _point_policy_payload(config: FixedCasimirConfig) -> dict[str, Any]:
    """Return only inputs that can change one certified microscopic point."""

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
    return payload


def _policy_fingerprint(config: FixedCasimirConfig) -> str:
    encoded = json.dumps(
        _point_policy_payload(config),
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
    ) -> None:
        if not isinstance(config, FixedCasimirConfig):
            raise TypeError("config must be a FixedCasimirConfig")
        self.config = config
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._runner = _run_transverse_certifier if runner is None else runner
        self._policy_fingerprint = _policy_fingerprint(config)
        self._entries: dict[str, Mapping[str, Any]] = {}
        self._q_by_entry: dict[str, tuple[str, str]] = {}
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        if self.cache_path is not None and self.cache_path.exists():
            self._load()

    @property
    def cached_point_count(self) -> int:
        return len(self._entries)

    @property
    def unique_q_count(self) -> int:
        return len(set(self._q_by_entry.values()))

    def count_new_q(self, q_model: np.ndarray) -> int:
        points = self._unique_points(q_model)
        cached_q = set(self._q_by_entry.values())
        return sum(q_key not in cached_q for q_key, _ in points)

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
        self.requested_q_evaluations += len(points)
        missing = [
            (q_key, q)
            for q_key, q in points
            if not self._q_is_complete(q_key)
        ]
        cache_hits = len(points) - len(missing)
        self.cache_hit_q_evaluations += cache_hits
        new_batches = 0

        if missing:
            labels = tuple(_stable_q_label(q_key) for q_key, _ in missing)
            q_values = np.asarray([q for _, q in missing], dtype=float)
            manifest = OuterQNodeManifest(
                labels=labels,
                q_model=q_values,
                grids={},
                labels_by_spec={},
            )
            with TemporaryDirectory(prefix="lno327-adaptive-point-batch-") as temporary:
                output = Path(temporary) / "certification.json"
                certification = self._runner(self.config, manifest, output)
            self.certification_batches += 1
            self.new_q_evaluations += len(missing)
            new_batches = 1
            self._consume_payload(
                certification.payload,
                labels=labels,
                q_keys=tuple(q_key for q_key, _ in missing),
            )
            self._save()

        rows: list[Mapping[str, Any]] = []
        unresolved: list[Mapping[str, Any]] = []
        for q_key, _ in points:
            label = _stable_q_label(q_key)
            for pairing in self.config.pairings:
                for n in self.config.matsubara_indices:
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
            new_q_count=len(missing),
            cache_hit_q_count=cache_hits,
            certification_batches=new_batches,
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

    def _q_is_complete(self, q_key: tuple[str, str]) -> bool:
        return all(
            _entry_key(pairing, n, q_key) in self._entries
            for pairing in self.config.pairings
            for n in self.config.matsubara_indices
        )

    def _consume_payload(
        self,
        payload: Mapping[str, Any],
        *,
        labels: tuple[str, ...],
        q_keys: tuple[tuple[str, str], ...],
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
            if pairing not in self.config.pairings or n not in self.config.matsubara_indices:
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
        if payload.get("schema") != _CACHE_SCHEMA:
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
            key = _entry_key(pairing, n, q_key)
            self._entries[key] = point
            self._q_by_entry[key] = q_key

    def _save(self) -> None:
        if self.cache_path is None:
            return
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
            "schema": _CACHE_SCHEMA,
            "policy_fingerprint": self._policy_fingerprint,
            "point_policy": _point_policy_payload(self.config),
            "entries": entries,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


__all__ = [
    "CertifiedOuterQProvider",
    "CertifiedPointBatch",
    "CertifiedPointCacheError",
    "certified_primary_logdet",
]
