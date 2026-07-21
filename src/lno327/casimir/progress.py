"""Non-authoritative runtime progress events for the certified Casimir route.

The progress layer observes existing controller decisions and provider calls.  It never
performs a microscopic evaluation, tail fit, or acceptance calculation of its own.
Reporter failures are therefore isolated from the scientific result path.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import math
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence


ProgressSink = Callable[[Mapping[str, Any]], None]
_PROGRESS_SINK: ContextVar[ProgressSink | None] = ContextVar(
    "lno327_casimir_progress_sink",
    default=None,
)


@contextmanager
def progress_context(sink: ProgressSink | None) -> Iterator[None]:
    """Install one observer for the current execution context."""

    token = _PROGRESS_SINK.set(sink)
    try:
        yield
    finally:
        _PROGRESS_SINK.reset(token)


def emit_progress(event: str, **fields: Any) -> None:
    """Emit one structured event without allowing reporting to alter science."""

    sink = _PROGRESS_SINK.get()
    if sink is None:
        return
    payload = {"event": str(event), **fields}
    try:
        sink(payload)
    except Exception:
        # Progress is deliberately non-authoritative.  Formal results and caches must
        # not change because a terminal, status reader, or progress filesystem failed.
        return


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_sequence(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    output: list[float] = []
    for item in value:
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError):
            return ()
        if not math.isfinite(numeric):
            return ()
        output.append(numeric)
    return tuple(output)


def _max_ratio(values: Any, tolerances: Any) -> float | None:
    numerator = _safe_sequence(values)
    denominator = _safe_sequence(tolerances)
    if not numerator or len(numerator) != len(denominator):
        return None
    ratios: list[float] = []
    for value, tolerance in zip(numerator, denominator, strict=True):
        if tolerance > 0.0:
            ratios.append(abs(value) / tolerance)
        elif value == 0.0:
            ratios.append(0.0)
        else:
            ratios.append(math.inf)
    return max(ratios, default=0.0)


def _reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("reason", "unresolved")).strip() or "unresolved"
        counts[reason] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _provider_statistics(provider: Any) -> dict[str, Any]:
    summary = getattr(provider, "performance_summary", None)
    if callable(summary):
        try:
            payload = summary()
        except Exception:
            payload = {}
        if isinstance(payload, Mapping):
            return dict(payload)
    names = (
        "cached_point_count",
        "unique_q_count",
        "certification_batches",
        "certification_failed_batches",
        "requested_q_evaluations",
        "new_q_evaluations",
        "cache_hit_q_evaluations",
        "requested_point_evaluations",
        "new_point_evaluations",
        "cache_hit_point_evaluations",
    )
    output: dict[str, Any] = {}
    for name in names:
        try:
            output[name] = int(getattr(provider, name, 0))
        except (TypeError, ValueError, OverflowError):
            output[name] = 0
    return output


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for name in set(before) | set(after):
        try:
            left = int(before.get(name, 0))
            right = int(after.get(name, 0))
        except (TypeError, ValueError, OverflowError):
            continue
        if right != left:
            output[name] = right - left
    return output


def _certifier_payload_summary(payload: Any) -> dict[str, Any]:
    rows = _safe_mapping(payload).get("point_results", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return {
            "established_point_count": 0,
            "unresolved_point_count": 0,
            "selected_N_distribution": {},
            "unresolved_reason_counts": {},
        }
    selected: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    established = 0
    unresolved = 0
    for raw in rows:
        point = _safe_mapping(raw)
        sweet = _safe_mapping(point.get("sweet_spot"))
        if sweet.get("status") == "established":
            established += 1
            try:
                selected[str(int(sweet["audit_N"]))] += 1
            except (KeyError, TypeError, ValueError, OverflowError):
                selected["unknown"] += 1
        else:
            unresolved += 1
            reason = (
                sweet.get("reason")
                or sweet.get("termination_reason")
                or point.get("reason")
                or point.get("termination_reason")
                or sweet.get("status")
                or "unresolved"
            )
            reasons[str(reason)] += 1
    return {
        "established_point_count": established,
        "unresolved_point_count": unresolved,
        "selected_N_distribution": dict(
            sorted(selected.items(), key=lambda item: item[0])
        ),
        "unresolved_reason_counts": dict(
            sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def pairing_budget_ratios(pairing_results: Any) -> dict[str, Any]:
    """Extract normalized error/budget ratios from already-computed result records."""

    if not isinstance(pairing_results, Mapping):
        return {}
    output: dict[str, Any] = {}
    for pairing, raw in pairing_results.items():
        record = _safe_mapping(raw)
        row: dict[str, Any] = {}
        candidates = {
            "radial": (
                "estimated_radial_errors_J_m2",
                "radial_tolerances_J_m2",
            ),
            "outer_finite": (
                "finite_domain_error_bounds_J_m2",
                "finite_domain_budget_tolerances_J_m2",
            ),
            "outer_tail": (
                "estimated_outer_tail_bounds_J_m2",
                "tail_budget_tolerances_J_m2",
            ),
        }
        for label, (value_name, tolerance_name) in candidates.items():
            ratio = _max_ratio(record.get(value_name), record.get(tolerance_name))
            if ratio is not None:
                row[label] = ratio
        scalar_candidates = {
            "matsubara_finite": (
                "finite_matsubara_outer_error_bound_J_m2",
                "finite_matsubara_budget_tolerance_J_m2",
            ),
            "matsubara_tail": (
                "estimated_matsubara_tail_bound_J_m2",
                "matsubara_tail_budget_tolerance_J_m2",
            ),
            "total": (
                "estimated_total_error_J_m2",
                "total_free_energy_tolerance_J_m2",
            ),
        }
        for label, (value_name, tolerance_name) in scalar_candidates.items():
            try:
                value = float(record[value_name])
                tolerance = float(record[tolerance_name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(value) or not math.isfinite(tolerance):
                continue
            row[label] = (
                abs(value) / tolerance
                if tolerance > 0.0
                else 0.0
                if value == 0.0
                else math.inf
            )
        for name in (
            "outer_tail_certificate_path",
            "matsubara_tail_certificate_path",
            "matsubara_tail_decay_passed",
            "matsubara_tail_holdout_passed",
            "finite_matsubara_budget_passed",
            "matsubara_tail_budget_passed",
            "total_free_energy_budget_passed",
        ):
            if name in record:
                row[name] = record[name]
        output[str(pairing)] = row
    return output


class ProgressPointProvider:
    """Observer wrapper around an existing certified-point provider."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        emit_progress(
            "microscopic_provider_initialized",
            provider_statistics=_provider_statistics(provider),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def reconfigure(self, config: Any) -> None:
        self._provider.reconfigure(config)
        emit_progress(
            "microscopic_provider_reconfigured",
            pairings=list(getattr(config, "pairings", ())),
            matsubara_indices=[
                int(value) for value in getattr(config, "matsubara_indices", ())
            ],
        )

    def performance_summary(self) -> dict[str, Any]:
        return _provider_statistics(self._provider)

    def evaluate(self, q_model: Any) -> Any:
        before = _provider_statistics(self._provider)
        try:
            requested_nodes = int(len(q_model))
        except (TypeError, ValueError, OverflowError):
            requested_nodes = 0
        emit_progress(
            "microscopic_request_started",
            requested_q_nodes=requested_nodes,
            provider_statistics=before,
        )
        try:
            result = self._provider.evaluate(q_model)
        except Exception as exc:
            emit_progress(
                "microscopic_request_failed",
                requested_q_nodes=requested_nodes,
                error_type=type(exc).__name__,
                error=str(exc),
                provider_statistics=_provider_statistics(self._provider),
            )
            raise
        after = _provider_statistics(self._provider)
        unresolved = tuple(
            row
            for row in getattr(result, "unresolved_points", ())
            if isinstance(row, Mapping)
        )
        emit_progress(
            "microscopic_request_completed",
            requested_q_count=int(getattr(result, "requested_q_count", requested_nodes)),
            new_q_count=int(getattr(result, "new_q_count", 0)),
            cache_hit_q_count=int(getattr(result, "cache_hit_q_count", 0)),
            requested_point_count=int(getattr(result, "requested_point_count", 0)),
            new_point_count=int(getattr(result, "new_point_count", 0)),
            cache_hit_point_count=int(getattr(result, "cache_hit_point_count", 0)),
            certification_batches=int(getattr(result, "certification_batches", 0)),
            all_established=bool(getattr(result, "all_established", False)),
            unresolved_point_count=len(unresolved),
            unresolved_reason_counts=_reason_counts(unresolved),
            provider_statistics=after,
            provider_counter_delta=_counter_delta(before, after),
        )
        return result


def wrap_certifier_runner(runner: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap one existing transverse certifier call with batch-level events."""

    def observed(config: Any, manifest: Any, output: Any) -> Any:
        labels = tuple(getattr(manifest, "labels", ()))
        pairings = tuple(str(value) for value in getattr(config, "pairings", ()))
        indices = tuple(int(value) for value in getattr(config, "matsubara_indices", ()))
        point_count = len(labels) * len(pairings) * len(indices)
        emit_progress(
            "microscopic_batch_started",
            requested_q_count=len(labels),
            requested_point_count=point_count,
            pairings=list(pairings),
            matsubara_indices=list(indices),
            N_candidates=[int(value) for value in getattr(config, "N_candidates", ())],
        )
        started = perf_counter()
        try:
            result = runner(config, manifest, output)
        except Exception as exc:
            emit_progress(
                "microscopic_batch_failed",
                requested_q_count=len(labels),
                requested_point_count=point_count,
                pairings=list(pairings),
                matsubara_indices=list(indices),
                wall_seconds=float(perf_counter() - started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        summary = _certifier_payload_summary(getattr(result, "payload", {}))
        emit_progress(
            "microscopic_batch_completed",
            requested_q_count=len(labels),
            requested_point_count=point_count,
            pairings=list(pairings),
            matsubara_indices=list(indices),
            wall_seconds=float(perf_counter() - started),
            **summary,
        )
        return result

    return observed


def run_progress_adaptive_joint(
    config: Any,
    *,
    provider: Any | None = None,
) -> Any:
    """Run the existing joint controller while observing each radial sub-run."""

    from .adaptive_joint_q import run_adaptive_joint_casimir
    from .adaptive_outer_q import run_adaptive_radial_casimir

    radial_run_index = 0
    emit_progress(
        "joint_controller_started",
        angular_orders=[int(value) for value in config.angular_orders],
        initial_radial_round_cap=int(config.initial_radial_round_cap),
        maximum_radial_round_cap=int(config.radial_config.max_refinement_rounds),
        u_max=float(config.radial_config.u_max),
    )

    def radial_runner(radial_config: Any, *, provider: Any | None = None) -> Any:
        nonlocal radial_run_index
        radial_run_index += 1
        emit_progress(
            "radial_run_started",
            radial_run_index=radial_run_index,
            angular_order=int(radial_config.angular_order),
            angular_offset_fraction=float(radial_config.angular_offset_fraction),
            radial_order=int(radial_config.radial_order),
            radial_round_cap=int(radial_config.max_refinement_rounds),
            initial_panel_count=len(radial_config.initial_panel_edges) - 1,
            u_max=float(radial_config.u_max),
        )
        started = perf_counter()
        try:
            result = run_adaptive_radial_casimir(
                radial_config,
                provider=provider,
            )
        except Exception as exc:
            emit_progress(
                "radial_run_failed",
                radial_run_index=radial_run_index,
                angular_order=int(radial_config.angular_order),
                radial_round_cap=int(radial_config.max_refinement_rounds),
                wall_seconds=float(perf_counter() - started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        emit_progress(
            "radial_run_completed",
            radial_run_index=radial_run_index,
            angular_order=int(radial_config.angular_order),
            angular_offset_fraction=float(radial_config.angular_offset_fraction),
            radial_round_cap=int(radial_config.max_refinement_rounds),
            status=str(result.status),
            radial_converged=bool(result.radial_converged),
            refinement_rounds=int(result.refinement_rounds),
            leaf_panel_count=len(result.panel_records),
            all_microscopic_nodes_certified=bool(
                result.all_microscopic_nodes_certified
            ),
            termination_reason=str(result.termination_reason),
            unresolved_reason_counts=_reason_counts(
                tuple(
                    row
                    for row in result.unresolved_points
                    if isinstance(row, Mapping)
                )
            ),
            budget_ratios=pairing_budget_ratios(result.pairing_results),
            provider_statistics=dict(result.provider_statistics),
            wall_seconds=float(perf_counter() - started),
        )
        return result

    started = perf_counter()
    try:
        result = run_adaptive_joint_casimir(
            config,
            provider=provider,
            radial_runner=radial_runner,
        )
    except Exception as exc:
        emit_progress(
            "joint_controller_failed",
            u_max=float(config.radial_config.u_max),
            wall_seconds=float(perf_counter() - started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    emit_progress(
        "joint_controller_completed",
        status=str(result.status),
        joint_converged=bool(result.joint_converged),
        radial_budget_passed=bool(result.radial_budget_passed),
        angular_budget_passed=bool(result.angular_budget_passed),
        offset_audit_passed=bool(result.offset_audit_passed),
        selected_angular_order=result.selected_angular_order,
        selected_radial_round_cap=result.selected_radial_round_cap,
        termination_reason=str(result.termination_reason),
        direction_record_count=len(result.direction_records),
        radial_run_count=len(result.radial_run_records),
        budget_ratios=pairing_budget_ratios(result.pairing_results),
        provider_statistics=dict(result.provider_statistics),
        wall_seconds=float(perf_counter() - started),
    )
    return result


def run_progress_outer_tail(
    config: Any,
    *,
    provider: Any | None = None,
) -> Any:
    """Observe every finite-domain cutoff used by the certified outer-tail controller."""

    from .certified_tail import run_certified_outer_tail_casimir

    cutoff_index = 0
    cutoff_values = tuple(float(value) for value in config.cutoff_u_values)
    emit_progress(
        "outer_controller_started",
        cutoff_u_values=list(cutoff_values),
        matsubara_indices=[
            int(value)
            for value in config.joint_config.radial_config.point_config.matsubara_indices
        ],
    )

    def joint_runner(joint_config: Any, *, provider: Any | None = None) -> Any:
        nonlocal cutoff_index
        cutoff_index += 1
        u_max = float(joint_config.radial_config.u_max)
        emit_progress(
            "outer_cutoff_started",
            cutoff_index=cutoff_index,
            cutoff_count=len(cutoff_values),
            u_max=u_max,
            cutoff_u_values=list(cutoff_values),
        )
        started = perf_counter()
        try:
            result = run_progress_adaptive_joint(
                joint_config,
                provider=provider,
            )
        except Exception as exc:
            emit_progress(
                "outer_cutoff_failed",
                cutoff_index=cutoff_index,
                cutoff_count=len(cutoff_values),
                u_max=u_max,
                wall_seconds=float(perf_counter() - started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        emit_progress(
            "outer_cutoff_completed",
            cutoff_index=cutoff_index,
            cutoff_count=len(cutoff_values),
            u_max=u_max,
            finite_domain_status=str(result.status),
            finite_domain_converged=bool(result.joint_converged),
            all_microscopic_nodes_certified=bool(
                result.all_microscopic_nodes_certified
            ),
            selected_angular_order=result.selected_angular_order,
            selected_radial_round_cap=result.selected_radial_round_cap,
            termination_reason=str(result.termination_reason),
            budget_ratios=pairing_budget_ratios(result.pairing_results),
            provider_statistics=dict(result.provider_statistics),
            wall_seconds=float(perf_counter() - started),
        )
        return result

    started = perf_counter()
    try:
        result = run_certified_outer_tail_casimir(
            config,
            provider=provider,
            joint_runner=joint_runner,
        )
    except Exception as exc:
        emit_progress(
            "outer_controller_failed",
            wall_seconds=float(perf_counter() - started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    emit_progress(
        "outer_controller_completed",
        status=str(result.status),
        cutoff_converged=bool(result.cutoff_converged),
        outer_tail_estimated=bool(result.outer_tail_estimated_flag),
        all_microscopic_nodes_certified=bool(
            result.all_microscopic_nodes_certified
        ),
        selected_u_max=result.selected_u_max,
        termination_reason=str(result.termination_reason),
        budget_ratios=pairing_budget_ratios(result.pairing_results),
        provider_statistics=dict(result.provider_statistics),
        wall_seconds=float(perf_counter() - started),
    )
    return result


class ProgressMatsubaraOuterRunner:
    """Callable observing one cumulative Matsubara block at a time."""

    def __init__(self, cutoff_values: Sequence[int]) -> None:
        self.cutoff_values = tuple(int(value) for value in cutoff_values)
        self.previous_cutoff = -1
        self.position = 0

    def __call__(self, config: Any, *, provider: Any | None = None) -> Any:
        self.position += 1
        point = config.joint_config.radial_config.point_config
        indices = tuple(int(value) for value in point.matsubara_indices)
        cutoff = max(indices)
        left = self.previous_cutoff + 1
        emit_progress(
            "matsubara_block_started",
            block_index=self.position,
            block_count=len(self.cutoff_values),
            left_n=left,
            right_n=cutoff,
            cumulative_term_count=len(indices),
            cutoff_values=list(self.cutoff_values),
        )
        started = perf_counter()
        try:
            result = run_progress_outer_tail(config, provider=provider)
        except Exception as exc:
            emit_progress(
                "matsubara_block_failed",
                block_index=self.position,
                block_count=len(self.cutoff_values),
                left_n=left,
                right_n=cutoff,
                wall_seconds=float(perf_counter() - started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.previous_cutoff = cutoff
        emit_progress(
            "matsubara_block_completed",
            block_index=self.position,
            block_count=len(self.cutoff_values),
            left_n=left,
            right_n=cutoff,
            outer_status=str(result.status),
            outer_cutoff_converged=bool(result.cutoff_converged),
            outer_tail_estimated=bool(result.outer_tail_estimated_flag),
            selected_u_max=result.selected_u_max,
            all_microscopic_nodes_certified=bool(
                result.all_microscopic_nodes_certified
            ),
            termination_reason=str(result.termination_reason),
            budget_ratios=pairing_budget_ratios(result.pairing_results),
            provider_statistics=dict(result.provider_statistics),
            wall_seconds=float(perf_counter() - started),
        )
        return result


__all__ = [
    "ProgressMatsubaraOuterRunner",
    "ProgressPointProvider",
    "ProgressSink",
    "emit_progress",
    "pairing_budget_ratios",
    "progress_context",
    "run_progress_adaptive_joint",
    "run_progress_outer_tail",
    "wrap_certifier_runner",
]
