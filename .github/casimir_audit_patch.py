from pathlib import Path


def replace_one(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if old not in text:
        raise SystemExit(f'missing expected text in {path}: {old[:120]!r}')
    if text.count(old) != 1:
        raise SystemExit(f'expected one match in {path}, found {text.count(old)}')
    p.write_text(text.replace(old, new), encoding='utf-8')


# 1. Robust CLI float serialization for negative scientific notation.
replace_one(
    'src/lno327/casimir/fixed_chain.py',
    'def _transverse_certification_command(\n',
    '''def _cli_float(value: float) -> str:\n    """Return an argparse-safe round-trip decimal without exponent notation."""\n\n    converted = float(value)\n    if not np.isfinite(converted):\n        raise ValueError("CLI float arguments must be finite")\n    return np.format_float_positional(converted, unique=True, trim="-")\n\n\ndef _transverse_certification_command(\n''',
)
replace_one(
    'src/lno327/casimir/fixed_chain.py',
    '        command.extend(["--q-point", label, repr(float(q[0])), repr(float(q[1]))])\n',
    '        command.extend(["--q-point", label, _cli_float(q[0]), _cli_float(q[1])])\n',
)
replace_one(
    'src/lno327/casimir/fixed_chain.py',
    '            repr(config.plate_angles_deg[0]),\n            repr(config.plate_angles_deg[1]),\n',
    '            _cli_float(config.plate_angles_deg[0]),\n            _cli_float(config.plate_angles_deg[1]),\n',
)

# 2. Expose the calibrated logdet policy through the public builder and CLI.
replace_one(
    'src/lno327/casimir/production.py',
    '    required_consecutive_passes: int = 2,\n    workers: int = 0,\n',
    '    required_consecutive_passes: int = 2,\n    logdet_rtol: float = 1e-3,\n    logdet_atol: float = 1e-6,\n    workers: int = 0,\n',
)
replace_one(
    'src/lno327/casimir/production.py',
    '        required_consecutive_passes=int(required_consecutive_passes),\n        workers=int(workers),\n',
    '        required_consecutive_passes=int(required_consecutive_passes),\n        logdet_rtol=float(logdet_rtol),\n        logdet_atol=float(logdet_atol),\n        workers=int(workers),\n',
)
replace_one(
    'src/lno327/casimir/cli.py',
    '    parser.add_argument("--workers", type=int, default=0)\n',
    '    parser.add_argument("--logdet-rtol", type=float, default=1e-3)\n    parser.add_argument("--logdet-atol", type=float, default=1e-6)\n    parser.add_argument("--workers", type=int, default=0)\n',
)
replace_one(
    'src/lno327/casimir/cli.py',
    '        plate_angles_deg=tuple(args.plate_angles_deg),\n        workers=args.workers,\n',
    '        plate_angles_deg=tuple(args.plate_angles_deg),\n        logdet_rtol=args.logdet_rtol,\n        logdet_atol=args.logdet_atol,\n        workers=args.workers,\n',
)

# 3. Preserve prefetch unresolved details and correct exception ordering/state.
replace_one(
    'src/lno327/casimir/adaptive_joint_q.py',
    '''            batch = evaluate(combined)\n            if not batch.all_established:\n                raise FixedCasimirExecutionError(\n                    "prefetched comparison contains unresolved microscopic points"\n                )\n''',
    '''            batch = evaluate(combined)\n            if not batch.all_established:\n                samples = [dict(value) for value in batch.unresolved_points[:8]]\n                raise FixedCasimirExecutionError(\n                    "prefetched comparison contains unresolved microscopic points: "\n                    + json.dumps(samples, sort_keys=True)\n                )\n''',
)
replace_one(
    'src/lno327/casimir/adaptive_joint_q.py',
    'from dataclasses import dataclass, replace\n',
    'from dataclasses import dataclass, replace\nimport json\n',
)
replace_one(
    'src/lno327/casimir/adaptive_joint_q.py',
    '''    except RuntimeError as exc:\n        reason = (\n            "joint_microscopic_q_node_budget_exhausted"\n            if str(exc) == "joint_microscopic_q_node_budget_exhausted"\n            else f"joint_runtime_failure: {exc}"\n        )\n        return _unresolved_result(\n            config,\n            direction_records=direction_records,\n            radial_run_records=radial_run_records,\n            offset_record=None,\n            selected_order=config.angular_orders[current_index],\n            radial_round_cap=radial_round_cap,\n            pairing_results=last_pairing_results,\n            radial_passed=last_radial_passed,\n            angular_passed=last_angular_passed,\n            offset_passed=last_offset_passed,\n            all_certified=all_certified,\n            reason=reason,\n            provider=active_provider,\n        )\n    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:\n        return _unresolved_result(\n            config,\n            direction_records=direction_records,\n            radial_run_records=radial_run_records,\n            offset_record=None,\n            selected_order=config.angular_orders[current_index],\n            radial_round_cap=radial_round_cap,\n            pairing_results=last_pairing_results,\n            radial_passed=False,\n            angular_passed=False,\n            offset_passed=False,\n            all_certified=False,\n            reason=f"point_provider_failure: {exc}",\n            provider=active_provider,\n        )\n''',
    '''    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:\n        return _unresolved_result(\n            config,\n            direction_records=direction_records,\n            radial_run_records=radial_run_records,\n            offset_record=None,\n            selected_order=config.angular_orders[current_index],\n            radial_round_cap=radial_round_cap,\n            pairing_results=last_pairing_results,\n            radial_passed=False,\n            angular_passed=False,\n            offset_passed=False,\n            all_certified=False,\n            reason=f"point_provider_failure: {exc}",\n            provider=active_provider,\n        )\n    except RuntimeError as exc:\n        reason = (\n            "joint_microscopic_q_node_budget_exhausted"\n            if str(exc) == "joint_microscopic_q_node_budget_exhausted"\n            else f"joint_runtime_failure: {exc}"\n        )\n        return _unresolved_result(\n            config,\n            direction_records=direction_records,\n            radial_run_records=radial_run_records,\n            offset_record=None,\n            selected_order=config.angular_orders[current_index],\n            radial_round_cap=radial_round_cap,\n            pairing_results=last_pairing_results,\n            radial_passed=last_radial_passed,\n            angular_passed=last_angular_passed,\n            offset_passed=last_offset_passed,\n            all_certified=all_certified,\n            reason=reason,\n            provider=active_provider,\n        )\n''',
)

# 4. Avoid duplicate/full pretty cache rewrites while preserving per-chunk checkpoints.
replace_one(
    'src/lno327/casimir/certified_point_provider.py',
    '        new_point_count = 0\n        new_batch_count = 0\n',
    '        new_point_count = 0\n        new_batch_count = 0\n        cache_dirty = False\n',
)
replace_one(
    'src/lno327/casimir/certified_point_provider.py',
    '''                self.certification_batches += 1\n                self.new_q_evaluations += len(chunk)\n''',
    '''                self.certification_batches += 1\n                cache_dirty = True\n                self.new_q_evaluations += len(chunk)\n''',
)
replace_one(
    'src/lno327/casimir/certified_point_provider.py',
    '''                if (\n                    self.cache_path is not None\n                    and len(group) > self.certifier_q_batch_size\n                ):\n                    self._save()\n\n        cache_hit_point_count = requested_point_count - new_point_count\n        self.cache_hit_point_evaluations += cache_hit_point_count\n        if groups:\n            self._save()\n''',
    '''                if (\n                    self.cache_path is not None\n                    and len(group) > self.certifier_q_batch_size\n                ):\n                    self._save()\n                    cache_dirty = False\n\n        cache_hit_point_count = requested_point_count - new_point_count\n        self.cache_hit_point_evaluations += cache_hit_point_count\n        if cache_dirty:\n            self._save()\n''',
)
replace_one(
    'src/lno327/casimir/certified_point_provider.py',
    '''        temporary.write_text(\n            json.dumps(payload, sort_keys=True, indent=2) + "\\n",\n            encoding="utf-8",\n        )\n''',
    '''        temporary.write_text(\n            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\\n",\n            encoding="utf-8",\n        )\n''',
)

# 5. Script defaults, profile validation, and resource headroom.
replace_one(
    'scripts/full_casimir/config.py',
    'DEFAULT_RTOL = 5e-3\nDEFAULT_ATOL_J_M2 = 1e-12\n',
    'DEFAULT_RTOL = 5e-3\nDEFAULT_ATOL_J_M2 = 1e-12\nDEFAULT_LOGDET_RTOL = 1.5e-3\nDEFAULT_LOGDET_ATOL = 1e-6\n',
)
replace_one(
    'scripts/full_casimir/config.py',
    'DEFAULT_RESERVED_LOGICAL_CPUS = 4\nDEFAULT_WORKER_CAP = 28\n',
    'DEFAULT_RESERVED_LOGICAL_CPUS = 6\nDEFAULT_WORKER_CAP = 26\n',
)
replace_one(
    'scripts/full_casimir/config.py',
    'PROFILE_NAME = "runtime_budget_v2"\n',
    'PROFILE_NAME = "runtime_budget_v3"\n',
)
replace_one(
    'scripts/full_casimir/config.py',
    'def validate_pairings(values: Iterable[str]) -> tuple[str, ...]:\n',
    '''def validate_profile(profile: str) -> str:\n    value = str(profile)\n    if not value or any(not (char.isalnum() or char in "._-") for char in value):\n        raise ValueError("profile must contain only letters, digits, '.', '_' or '-'")\n    return value\n\n\ndef validate_pairings(values: Iterable[str]) -> tuple[str, ...]:\n''',
)

# 6. Energy runner state handling, explicit unresolved exit status, calibrated policy.
replace_one(
    'scripts/full_casimir/energy.py',
    '    DEFAULT_LOG_ROOT,\n',
    '    DEFAULT_LOGDET_ATOL,\n    DEFAULT_LOGDET_RTOL,\n    DEFAULT_LOG_ROOT,\n',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '    rtol: float = DEFAULT_RTOL\n    atol_J_m2: float = DEFAULT_ATOL_J_M2\n',
    '    rtol: float = DEFAULT_RTOL\n    atol_J_m2: float = DEFAULT_ATOL_J_M2\n    logdet_rtol: float = DEFAULT_LOGDET_RTOL\n    logdet_atol: float = DEFAULT_LOGDET_ATOL\n',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '''def _case_state(run_dir: Path) -> str:\n    summary = _read_json(run_dir / "summary.json")\n    if summary:\n        return str(summary.get("status", "result_present"))\n    manifest = _read_json(run_dir / "manifest.json")\n    if manifest:\n        return str(manifest.get("status", "directory_present"))\n    return "missing"\n''',
    '''def _case_state(run_dir: Path) -> str:\n    manifest = _read_json(run_dir / "manifest.json")\n    summary = _read_json(run_dir / "summary.json")\n    manifest_status = str(manifest.get("status", ""))\n    if manifest_status in {"completed", "unresolved", "failed", "running"}:\n        return manifest_status\n    if bool(summary.get("matsubara_converged", False)):\n        return "completed"\n    if summary:\n        return "unresolved"\n    return "missing"\n''',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '    engineering_failures = 0\n',
    '    engineering_failures = 0\n    unresolved_results = 0\n',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '''                    required_consecutive_passes=options.required_consecutive_passes,\n                    workers=resources.workers,\n''',
    '''                    required_consecutive_passes=options.required_consecutive_passes,\n                    logdet_rtol=options.logdet_rtol,\n                    logdet_atol=options.logdet_atol,\n                    workers=resources.workers,\n''',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '''                else:\n                    print(\n                        f"UNRESOLVED: {case}: {result.termination_reason}",\n                        flush=True,\n                    )\n            except BaseException as exc:\n                error = exc\n                engineering_failures += 1\n''',
    '''                else:\n                    unresolved_results += 1\n                    print(\n                        f"UNRESOLVED: {case}: {result.termination_reason}",\n                        flush=True,\n                    )\n            except KeyboardInterrupt:\n                raise\n            except Exception as exc:\n                error = exc\n                engineering_failures += 1\n''',
)
replace_one(
    'scripts/full_casimir/energy.py',
    '''            if isinstance(error, KeyboardInterrupt):\n                raise error\n\n    return 1 if engineering_failures else 0\n''',
    '''    if engineering_failures:\n        return 1\n    return 2 if unresolved_results else 0\n''',
)

# 7. Workflow: apply thread environment before importing lno327, add logdet args and v3 profile.
replace_one(
    'scripts/full_casimir/workflow.py',
    'from .cleanup_legacy_root import cleanup_legacy_root_scripts\n',
    'from .cleanup_legacy_root import cleanup_legacy_root_scripts\nfrom .config import apply_single_thread_environment\n\napply_single_thread_environment()\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '    DEFAULT_LOG_ROOT,\n',
    '    DEFAULT_LOGDET_ATOL,\n    DEFAULT_LOGDET_RTOL,\n    DEFAULT_LOG_ROOT,\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '    validate_pairings,\n',
    '    validate_pairings,\n    validate_profile,\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    'PILOT_PROFILE = "0deg_pilot_v2"\n',
    'PILOT_PROFILE = "0deg_pilot_v3"\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '        help="Logical CPUs kept free for desktop responsiveness (default: 4).",\n',
    '        help="Logical CPUs kept free for desktop responsiveness (default: 6).",\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '        help="Maximum worker processes (default: 28).",\n',
    '        help="Maximum worker processes (default: 26).",\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)\n    parser.add_argument("--atol-J-m2", type=float, default=DEFAULT_ATOL_J_M2)\n',
    '    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)\n    parser.add_argument("--atol-J-m2", type=float, default=DEFAULT_ATOL_J_M2)\n    parser.add_argument("--logdet-rtol", type=float, default=DEFAULT_LOGDET_RTOL)\n    parser.add_argument("--logdet-atol", type=float, default=DEFAULT_LOGDET_ATOL)\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '        atol_J_m2=float(args.atol_J_m2),\n',
    '        atol_J_m2=float(args.atol_J_m2),\n        logdet_rtol=float(args.logdet_rtol),\n        logdet_atol=float(args.logdet_atol),\n',
)
replace_one(
    'scripts/full_casimir/workflow.py',
    '''    profile = args.profile or (\n        PILOT_PROFILE if args.command == "pilots" else PROFILE_NAME\n    )\n''',
    '''    profile = validate_profile(\n        args.profile or (PILOT_PROFILE if args.command == "pilots" else PROFILE_NAME)\n    )\n''',
)

# 8. Background launcher: set threads before Python import, preserve logs, reject stale PID reuse.
replace_one(
    'scripts/full_casimir/background.sh',
    'mkdir -p "$LOG_ROOT"\n',
    '''mkdir -p "$LOG_ROOT"\n\nexport OMP_NUM_THREADS=1\nexport OPENBLAS_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\nexport NUMEXPR_NUM_THREADS=1\nexport BLIS_NUM_THREADS=1\nexport VECLIB_MAXIMUM_THREADS=1\nexport OMP_DYNAMIC=FALSE\nexport MKL_DYNAMIC=FALSE\nexport MALLOC_ARENA_MAX=4\nexport PYTHONUNBUFFERED=1\n''',
)
replace_one(
    'scripts/full_casimir/background.sh',
    '''    pid="$(cat "$PID_FILE")"\n    kill -0 "$pid" 2>/dev/null\n''',
    '''    pid="$(cat "$PID_FILE")"\n    [[ "$pid" =~ ^[0-9]+$ ]] || return 1\n    kill -0 "$pid" 2>/dev/null || return 1\n    local cmdline\n    cmdline="$(tr '\\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"\n    [[ "$cmdline" == *"scripts.full_casimir.workflow"* ]]\n''',
)
replace_one(
    'scripts/full_casimir/background.sh',
    '    : > "$DRIVER_LOG"\n',
    '    printf "\\n===== %s start mode=%s =====\\n" "$(date --iso-8601=seconds)" "$mode" >> "$DRIVER_LOG"\n',
)
replace_one(
    'scripts/full_casimir/background.sh',
    '            echo "last PID: $(cat "$PID_FILE")"\n',
    '            echo "stale/last PID: $(cat "$PID_FILE")"\n',
)
replace_one(
    'scripts/full_casimir/background.sh',
    '            echo "stopped"\n            return 0\n',
    '            rm -f "$PID_FILE"\n            echo "stopped"\n            return 0\n',
)

# 9. Torque metadata accurately scopes propagated numerical error.
replace_one(
    'scripts/full_casimir/postprocess.py',
    '        "torque_per_area_unit": "N/m",\n',
    '        "torque_per_area_unit": "N/m",\n        "torque_error_scope": "worst_case_propagation_of_input_energy_numerical_bounds_only",\n        "finite_difference_truncation_error_bounded": False,\n',
)

# 10. Add focused regression tests.
Path('tests/test_casimir_pilot_v3_audit.py').write_text(r'''from __future__ import annotations

from pathlib import Path

import numpy as np

from lno327.casimir.adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    run_adaptive_joint_casimir,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig, _cli_float
from lno327.casimir.production import build_full_casimir_config
from scripts.full_casimir.config import (
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_RESERVED_LOGICAL_CPUS,
    DEFAULT_WORKER_CAP,
    PROFILE_NAME,
    validate_profile,
)
from scripts.full_casimir.energy import _case_state


def test_cli_float_avoids_exponent_for_negative_tiny_values() -> None:
    token = _cli_float(-1.25e-17)
    assert token.startswith('-0.')
    assert 'e' not in token.lower()
    assert float(token) == -1.25e-17


def test_public_builder_accepts_calibrated_logdet_policy() -> None:
    config = build_full_casimir_config(
        pairings=('dwave',),
        logdet_rtol=1.5e-3,
        logdet_atol=1e-6,
    )
    point = config.outer_tail_config.joint_config.radial_config.point_config
    assert point.logdet_rtol == 1.5e-3
    assert point.logdet_atol == 1e-6


def test_prefetch_unresolved_is_provider_failure_and_not_certified() -> None:
    class Provider:
        unique_q_count = 0

        def count_new_q(self, q_model: np.ndarray) -> int:
            return len(q_model)

        def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch:
            return CertifiedPointBatch(
                point_results=(),
                unresolved_points=(
                    {
                        'pairing': 'spm',
                        'n': 1,
                        'q_label': 'q0',
                        'reason': 'point sweet spot is not established',
                    },
                ),
                requested_q_count=len(q_model),
                new_q_count=len(q_model),
                cache_hit_q_count=0,
                certification_batches=0,
            )

        def performance_statistics(self):
            return {}

    result = run_adaptive_joint_casimir(
        AdaptiveJointCasimirConfig(),
        provider=Provider(),
    )
    assert result.status == 'unresolved'
    assert result.all_microscopic_nodes_certified is False
    assert result.termination_reason.startswith('point_provider_failure:')
    assert 'point sweet spot is not established' in result.termination_reason


def test_completed_manifest_wins_over_summary_status(tmp_path: Path) -> None:
    run_dir = tmp_path / 'case'
    run_dir.mkdir()
    (run_dir / 'manifest.json').write_text('{"status":"completed"}\n')
    (run_dir / 'summary.json').write_text(
        '{"status":"adaptive_tail_bounded","matsubara_converged":true}\n'
    )
    assert _case_state(run_dir) == 'completed'


def test_v3_script_defaults_and_profile_validation() -> None:
    assert DEFAULT_LOGDET_RTOL == 1.5e-3
    assert DEFAULT_LOGDET_ATOL == 1e-6
    assert DEFAULT_RESERVED_LOGICAL_CPUS == 6
    assert DEFAULT_WORKER_CAP == 26
    assert PROFILE_NAME == 'runtime_budget_v3'
    assert validate_profile('0deg_pilot_v3') == '0deg_pilot_v3'
    for invalid in ('', '../escape', 'bad profile', 'a/b'):
        try:
            validate_profile(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'profile should be rejected: {invalid!r}')


def test_fixed_config_retains_two_consecutive_passes() -> None:
    assert FixedCasimirConfig().required_consecutive_passes == 2
''', encoding='utf-8')
