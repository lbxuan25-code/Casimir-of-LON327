"""Frozen formal policy for arbitrary-q performance and numerical qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

FORMAL_POLICY_ID = "ArbitraryQFormalPolicyV2"
PERFORMANCE_WORKLOAD_ID = "ArbitraryQPerformanceWorkloadV2"
QUALIFICATION_MATRIX_ID = "ArbitraryQQualificationMatrixV2"
OUTER_Q_BATCH_WORKLOAD_ID = "outer_q_batch_v2"
QUALIFICATION_PRIMARY_WORKLOAD_ID = "qualification_primary_v2"
QUALIFICATION_AUDIT_WORKLOAD_ID = "qualification_audit_v2"
MODEL_WORKLOAD_ID = "symmetry_bdg_2band_bond_endpoint_gauge_v1"
PRIMITIVE_CONTRACT_VERSION = "ArbitraryQPrimitiveContract-v3"
MATERIAL_CACHE_SCHEMA = "MaterialGridCache-v3"
RESPONSE_CACHE_SCHEMA = "CrystalResponseCache-v3"
EXECUTION_STRATEGY = "persistent_fork_q_lab_angle_batch_tasks_ordered_parent_collection"
THREAD_POLICY_ID = "single_thread_blas_omp_v1"
SUPPORTED_Q_COMPONENT_LIMIT = float(np.pi)
# Compatibility alias.  This is a syntactic support limit, not a numerically
# qualified outer-integration envelope.
VALIDATED_Q_COMPONENT_LIMIT = SUPPORTED_Q_COMPONENT_LIMIT

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
        _jsonable(dict(config)), sort_keys=True, separators=(",", ":")
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
            raise ValueError(
                f"{self.policy_id} {self.section} policy failed: "
                + "; ".join(self.violations)
            )


def _set(values: Sequence[Any] | set[Any] | frozenset[Any]) -> set[Any]:
    return set(values)


def _fixed_float(
    config: Mapping[str, Any],
    key: str,
    expected: float,
    violations: list[str],
) -> None:
    value = float(config.get(key, np.nan))
    if not np.isfinite(value) or value != float(expected):
        violations.append(f"{key} must equal {expected!r}")


def _maximum_float(
    config: Mapping[str, Any],
    key: str,
    maximum: float,
    violations: list[str],
) -> None:
    value = float(config.get(key, np.inf))
    if not np.isfinite(value) or value > float(maximum) or value < 0.0:
        violations.append(f"{key} must be finite, non-negative and at most {maximum!r}")


def validate_performance_formal_config(
    config: Mapping[str, Any],
) -> FormalPolicyValidation:
    pairings = _set(config.get("pairings", ()))
    matsubara = {int(v) for v in config.get("matsubara_indices", ())}
    runtime_chunks = {int(v) for v in config.get("runtime_chunk_sizes", ())}
    violations: list[str] = []

    if str(config.get("performance_workload_id", "")) != PERFORMANCE_WORKLOAD_ID:
        violations.append(f"performance_workload_id must be {PERFORMANCE_WORKLOAD_ID!r}")
    if str(config.get("model_workload_id", "")) != MODEL_WORKLOAD_ID:
        violations.append(f"model_workload_id must be {MODEL_WORKLOAD_ID!r}")
    if not _PERFORMANCE_REQUIRED_PAIRINGS.issubset(pairings):
        violations.append("pairings must include spm and dwave")
    if int(config.get("N", 0)) < 128:
        violations.append("N must be at least 128")
    if int(config.get("q_tasks", 0)) < 8:
        violations.append("q_tasks must be at least 8")
    if int(config.get("workers", 0)) < 4:
        violations.append("workers must be at least 4")
    if int(config.get("qualification_primary_tasks", 0)) != 4:
        violations.append("qualification_primary_tasks must equal 4")
    if int(config.get("qualification_primary_workers", 0)) != 4:
        violations.append("qualification_primary_workers must equal 4")
    if int(config.get("qualification_audit_tasks", 0)) != 1:
        violations.append("qualification_audit_tasks must equal 1")
    if int(config.get("qualification_audit_workers", 0)) != 1:
        violations.append("qualification_audit_workers must equal 1")
    expected_workloads = {
        OUTER_Q_BATCH_WORKLOAD_ID,
        QUALIFICATION_PRIMARY_WORKLOAD_ID,
        QUALIFICATION_AUDIT_WORKLOAD_ID,
    }
    if not expected_workloads.issubset(_set(config.get("workload_classes", ()))):
        violations.append("workload_classes must include outer, qualification-primary and audit")
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
    _maximum_float(config, "comparison_atol", 2e-12, violations)
    _maximum_float(config, "comparison_rtol", 2e-11, violations)
    _fixed_float(config, "temperature_K", 10.0, violations)
    _fixed_float(config, "delta0_eV", 0.1, violations)
    _fixed_float(config, "eta_eV", 1e-8, violations)
    if str(config.get("execution_strategy", EXECUTION_STRATEGY)) != EXECUTION_STRATEGY:
        violations.append(f"execution_strategy must be {EXECUTION_STRATEGY!r}")
    if str(config.get("thread_policy_id", THREAD_POLICY_ID)) != THREAD_POLICY_ID:
        violations.append(f"thread_policy_id must be {THREAD_POLICY_ID!r}")

    return FormalPolicyValidation(
        policy_id=FORMAL_POLICY_ID,
        section="performance",
        passed=not violations,
        violations=tuple(violations),
        config_fingerprint=config_fingerprint(config),
    )


def validate_numerical_formal_config(
    config: Mapping[str, Any],
) -> FormalPolicyValidation:
    pairings = _set(config.get("pairings", ()))
    n_values = {int(v) for v in config.get("N_values", ())}
    matsubara = {int(v) for v in config.get("matsubara_indices", ())}
    violations: list[str] = []

    if str(config.get("qualification_matrix_id", "")) != QUALIFICATION_MATRIX_ID:
        violations.append(f"qualification_matrix_id must be {QUALIFICATION_MATRIX_ID!r}")
    if str(config.get("model_workload_id", "")) != MODEL_WORKLOAD_ID:
        violations.append(f"model_workload_id must be {MODEL_WORKLOAD_ID!r}")
    if not _NUMERICAL_REQUIRED_PAIRINGS.issubset(pairings):
        violations.append("pairings must include spm and dwave")
    if not _NUMERICAL_REQUIRED_N.issubset(n_values):
        violations.append("N_values must include 256,384,512")
    if int(config.get("reference_nk", 0)) != 1256:
        violations.append("reference_nk must equal 1256")
    if int(config.get("reference_order", 0)) < 384:
        violations.append("reference_order must be at least 384")
    if int(config.get("reference_panel_count", 0)) != 16:
        violations.append("reference_panel_count must equal 16")
    if int(config.get("reference_workers", 0)) != 8:
        violations.append("reference_workers must equal 8")
    if int(config.get("reference_task_size", 0)) != 4:
        violations.append("reference_task_size must equal 4")
    if not _NUMERICAL_REQUIRED_MATSUBARA.issubset(matsubara):
        violations.append("Matsubara indices must include 0,1,8")
    _maximum_float(config, "primitive_tolerance", 1e-3, violations)
    _maximum_float(config, "primitive_atol", 1e-12, violations)
    _maximum_float(config, "reflection_tolerance", 3e-4, violations)
    _maximum_float(config, "reflection_atol", 1e-12, violations)
    _maximum_float(config, "logdet_tolerance", 3e-4, violations)
    _maximum_float(config, "logdet_atol", 1e-14, violations)
    _maximum_float(config, "diagonal_observable_tolerance", 1e-3, violations)
    _maximum_float(config, "diagonal_observable_atol", 1e-12, violations)
    _maximum_float(config, "ward_tolerance", 1e-7, violations)
    _maximum_float(config, "ward_absolute_tolerance", 1e-12, violations)
    _fixed_float(config, "temperature_K", 10.0, violations)
    _fixed_float(config, "delta0_eV", 0.1, violations)
    _fixed_float(config, "eta_eV", 1e-8, violations)
    _fixed_float(config, "separation_nm", 20.0, violations)
    if int(config.get("canonical_block_size", 0)) != 4096:
        violations.append("canonical_block_size must equal 4096")
    if int(config.get("runtime_chunk_size", 0)) != 16384:
        violations.append("runtime_chunk_size must equal 16384")
    if int(config.get("primary_workers", 0)) != 4:
        violations.append("primary_workers must equal 4")
    if int(config.get("audit_workers", 0)) != 1:
        violations.append("audit_workers must equal 1")
    if str(config.get("execution_strategy", EXECUTION_STRATEGY)) != EXECUTION_STRATEGY:
        violations.append(f"execution_strategy must be {EXECUTION_STRATEGY!r}")
    if str(config.get("thread_policy_id", THREAD_POLICY_ID)) != THREAD_POLICY_ID:
        violations.append(f"thread_policy_id must be {THREAD_POLICY_ID!r}")

    return FormalPolicyValidation(
        policy_id=FORMAL_POLICY_ID,
        section="numerical",
        passed=not violations,
        violations=tuple(violations),
        config_fingerprint=config_fingerprint(config),
    )


def validate_performance_manifest_compatibility(
    *, manifest_config: Mapping[str, Any], qualification_config: Mapping[str, Any]
) -> tuple[str, ...]:
    violations: list[str] = []
    exact_pairs = (
        ("canonical_block_size", "canonical_block_size"),
        ("temperature_K", "temperature_K"),
        ("delta0_eV", "delta0_eV"),
        ("eta_eV", "eta_eV"),
        ("model_workload_id", "model_workload_id"),
        ("qualification_primary_workers", "primary_workers"),
        ("qualification_audit_workers", "audit_workers"),
        ("execution_strategy", "execution_strategy"),
        ("thread_policy_id", "thread_policy_id"),
    )
    for manifest_key, qualification_key in exact_pairs:
        if manifest_config.get(manifest_key) != qualification_config.get(qualification_key):
            violations.append(
                f"qualification {qualification_key} differs from performance {manifest_key}"
            )
    runtime_chunks = {int(v) for v in manifest_config.get("runtime_chunk_sizes", ())}
    if int(qualification_config.get("runtime_chunk_size", -1)) not in runtime_chunks:
        violations.append("qualification runtime chunk was not performance-qualified")
    required_classes = {
        QUALIFICATION_PRIMARY_WORKLOAD_ID,
        QUALIFICATION_AUDIT_WORKLOAD_ID,
    }
    if not required_classes.issubset(_set(manifest_config.get("workload_classes", ()))):
        violations.append("performance manifest lacks qualification workload classes")
    return tuple(violations)


def validate_q_domain(q_model: np.ndarray) -> np.ndarray:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    if np.any(np.abs(q) > SUPPORTED_Q_COMPONENT_LIMIT):
        raise ValueError(
            "q_model lies outside the syntactically supported principal domain "
            f"|q_i| <= pi: q={q.tolist()}"
        )
    return q


__all__ = [
    "EXECUTION_STRATEGY",
    "FORMAL_POLICY_ID",
    "FormalPolicyValidation",
    "MATERIAL_CACHE_SCHEMA",
    "MODEL_WORKLOAD_ID",
    "OUTER_Q_BATCH_WORKLOAD_ID",
    "PERFORMANCE_WORKLOAD_ID",
    "PRIMITIVE_CONTRACT_VERSION",
    "QUALIFICATION_AUDIT_WORKLOAD_ID",
    "QUALIFICATION_MATRIX_ID",
    "QUALIFICATION_PRIMARY_WORKLOAD_ID",
    "RESPONSE_CACHE_SCHEMA",
    "SUPPORTED_Q_COMPONENT_LIMIT",
    "THREAD_POLICY_ID",
    "VALIDATED_Q_COMPONENT_LIMIT",
    "config_fingerprint",
    "validate_numerical_formal_config",
    "validate_performance_formal_config",
    "validate_performance_manifest_compatibility",
    "validate_q_domain",
]
