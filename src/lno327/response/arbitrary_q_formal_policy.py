"""Frozen formal policy for arbitrary-q performance and numerical qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

FORMAL_POLICY_ID = "ArbitraryQFormalPolicyV1"
PRIMITIVE_CONTRACT_VERSION = "ArbitraryQPrimitiveContract-v2"
MATERIAL_CACHE_SCHEMA = "MaterialGridCache-v2"
RESPONSE_CACHE_SCHEMA = "CrystalResponseCache-v2"
EXECUTION_STRATEGY = "persistent_fork_q_lab_angle_batch_tasks_ordered_parent_collection"
THREAD_POLICY_ID = "single_thread_blas_omp_v1"
VALIDATED_Q_COMPONENT_LIMIT = float(np.pi)

_PERFORMANCE_REQUIRED_PAIRINGS = frozenset({"spm", "dwave"})
_PERFORMANCE_REQUIRED_MATSUBARA = frozenset({0, 1, 2, 4, 8})
_PERFORMANCE_REQUIRED_RUNTIME_CHUNKS = frozenset({4096, 16384})
_NUMERICAL_REQUIRED_PAIRINGS = frozenset({"spm", "dwave"})
_NUMERICAL_REQUIRED_N = frozenset({256, 384, 512})
_NUMERICAL_REQUIRED_MATSUBARA = frozenset({0, 1, 8})


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("formal policy fingerprints reject non-finite floats")
        return {"float_hex": (0.0 if value == 0.0 else value).hex()}
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=repr)
        return [_jsonable(item) for item in items]
    return repr(value)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FormalPolicyValidation:
    policy_id: str
    section: str
    passed: bool
    violations: tuple[str, ...]
    config_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "formal_policy_id": self.policy_id,
            "formal_policy_section": self.section,
            "formal_policy_passed": bool(self.passed),
            "formal_policy_violations": list(self.violations),
            "config_fingerprint": self.config_fingerprint,
        }

    def require_passed(self) -> None:
        if not self.passed:
            detail = "; ".join(self.violations)
            raise ValueError(
                f"{self.policy_id} {self.section} policy failed: {detail}"
            )


def _set(values: Sequence[Any] | set[Any] | frozenset[Any]) -> set[Any]:
    return set(values)


def validate_performance_formal_config(
    config: Mapping[str, Any],
) -> FormalPolicyValidation:
    pairings = _set(config.get("pairings", ()))
    matsubara = {int(v) for v in config.get("matsubara_indices", ())}
    runtime_chunks = {int(v) for v in config.get("runtime_chunk_sizes", ())}
    violations: list[str] = []

    if not _PERFORMANCE_REQUIRED_PAIRINGS.issubset(pairings):
        violations.append("pairings must include spm and dwave")
    if int(config.get("N", 0)) < 128:
        violations.append("N must be at least 128")
    if int(config.get("q_tasks", 0)) < 8:
        violations.append("q_tasks must be at least 8")
    if int(config.get("workers", 0)) < 4:
        violations.append("workers must be at least 4")
    if not _PERFORMANCE_REQUIRED_MATSUBARA.issubset(matsubara):
        violations.append("Matsubara indices must include 0,1,2,4,8")
    if int(config.get("canonical_block_size", 0)) != 4096:
        violations.append("canonical_block_size must equal 4096")
    if not _PERFORMANCE_REQUIRED_RUNTIME_CHUNKS.issubset(runtime_chunks):
        violations.append("runtime_chunk_sizes must include 4096 and 16384")
    if float(config.get("minimum_speedup", -np.inf)) < 4.0:
        violations.append("minimum_speedup must be at least 4")
    if float(config.get("minimum_cpu_wall_ratio", -np.inf)) < 4.0:
        violations.append("minimum_cpu_wall_ratio must be at least 4")
    if float(config.get("maximum_pool_overhead_fraction", np.inf)) > 0.05:
        violations.append("maximum_pool_overhead_fraction must be at most 0.05")
    strategy = str(config.get("execution_strategy", EXECUTION_STRATEGY))
    if strategy != EXECUTION_STRATEGY:
        violations.append(f"execution_strategy must be {EXECUTION_STRATEGY!r}")
    if str(config.get("thread_policy_id", THREAD_POLICY_ID)) != THREAD_POLICY_ID:
        violations.append(f"thread_policy_id must be {THREAD_POLICY_ID!r}")

    fingerprint = config_fingerprint(config)
    return FormalPolicyValidation(
        policy_id=FORMAL_POLICY_ID,
        section="performance",
        passed=not violations,
        violations=tuple(violations),
        config_fingerprint=fingerprint,
    )


def validate_numerical_formal_config(
    config: Mapping[str, Any],
) -> FormalPolicyValidation:
    pairings = _set(config.get("pairings", ()))
    n_values = {int(v) for v in config.get("N_values", ())}
    matsubara = {int(v) for v in config.get("matsubara_indices", ())}
    violations: list[str] = []

    if not _NUMERICAL_REQUIRED_PAIRINGS.issubset(pairings):
        violations.append("pairings must include spm and dwave")
    if not _NUMERICAL_REQUIRED_N.issubset(n_values):
        violations.append("N_values must include 256,384,512")
    if int(config.get("reference_nk", 0)) != 1256:
        violations.append("reference_nk must equal 1256")
    if int(config.get("reference_order", 0)) < 384:
        violations.append("reference_order must be at least 384")
    if not _NUMERICAL_REQUIRED_MATSUBARA.issubset(matsubara):
        violations.append("Matsubara indices must include 0,1,8")
    if float(config.get("primitive_tolerance", np.inf)) > 1e-3:
        violations.append("primitive_tolerance must be at most 1e-3")
    if float(config.get("reflection_tolerance", np.inf)) > 3e-4:
        violations.append("reflection_tolerance must be at most 3e-4")
    if float(config.get("logdet_tolerance", np.inf)) > 3e-4:
        violations.append("logdet_tolerance must be at most 3e-4")
    if float(config.get("diagonal_observable_tolerance", np.inf)) > 1e-3:
        violations.append("diagonal_observable_tolerance must be at most 1e-3")
    if int(config.get("canonical_block_size", 0)) != 4096:
        violations.append("canonical_block_size must equal 4096")
    if int(config.get("runtime_chunk_size", 0)) not in {4096, 16384}:
        violations.append("runtime_chunk_size must be 4096 or 16384")
    if int(config.get("workers", 0)) < 4:
        violations.append("workers must be at least 4")
    strategy = str(config.get("execution_strategy", EXECUTION_STRATEGY))
    if strategy != EXECUTION_STRATEGY:
        violations.append(f"execution_strategy must be {EXECUTION_STRATEGY!r}")
    if str(config.get("thread_policy_id", THREAD_POLICY_ID)) != THREAD_POLICY_ID:
        violations.append(f"thread_policy_id must be {THREAD_POLICY_ID!r}")

    fingerprint = config_fingerprint(config)
    return FormalPolicyValidation(
        policy_id=FORMAL_POLICY_ID,
        section="numerical",
        passed=not violations,
        violations=tuple(violations),
        config_fingerprint=fingerprint,
    )


def validate_performance_manifest_compatibility(
    *,
    manifest_config: Mapping[str, Any],
    qualification_config: Mapping[str, Any],
) -> tuple[str, ...]:
    violations: list[str] = []
    if int(manifest_config.get("canonical_block_size", 0)) != int(
        qualification_config.get("canonical_block_size", -1)
    ):
        violations.append("canonical reduction block differs from performance manifest")
    runtime_chunks = {int(v) for v in manifest_config.get("runtime_chunk_sizes", ())}
    if int(qualification_config.get("runtime_chunk_size", -1)) not in runtime_chunks:
        violations.append("qualification runtime chunk was not performance-qualified")
    if int(qualification_config.get("workers", 0)) != int(
        manifest_config.get("workers", -1)
    ):
        violations.append("qualification worker count differs from performance manifest")
    if str(manifest_config.get("execution_strategy", "")) != str(
        qualification_config.get("execution_strategy", "")
    ):
        violations.append("qualification execution strategy differs from performance manifest")
    if str(manifest_config.get("thread_policy_id", "")) != str(
        qualification_config.get("thread_policy_id", "")
    ):
        violations.append("qualification thread policy differs from performance manifest")
    return tuple(violations)


def validate_q_domain(q_model: np.ndarray) -> np.ndarray:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    if np.any(np.abs(q) > VALIDATED_Q_COMPONENT_LIMIT):
        raise ValueError(
            "q_model lies outside the validated microscopic principal domain "
            f"|q_i| <= pi: q={q.tolist()}"
        )
    return q


__all__ = [
    "EXECUTION_STRATEGY",
    "FORMAL_POLICY_ID",
    "FormalPolicyValidation",
    "MATERIAL_CACHE_SCHEMA",
    "PRIMITIVE_CONTRACT_VERSION",
    "RESPONSE_CACHE_SCHEMA",
    "THREAD_POLICY_ID",
    "VALIDATED_Q_COMPONENT_LIMIT",
    "config_fingerprint",
    "validate_numerical_formal_config",
    "validate_performance_formal_config",
    "validate_performance_manifest_compatibility",
    "validate_q_domain",
]
