from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "src/lno327/casimir/adaptive_joint_q.py"
replace_once(
    path,
    '''def _usable_radial_estimate(result: AdaptiveRadialCasimirResult) -> tuple[bool, str]:
    if not bool(result.all_microscopic_nodes_certified):
        reason = str(result.termination_reason)
        details = [
            str(record.get("reason", ""))
            for record in result.unresolved_points
            if isinstance(record, Mapping) and record.get("reason")
        ]
        if details and details[0] not in reason:
            reason = f"{reason}: {details[0]}"
        return False, reason
''',
    '''def _usable_radial_estimate(result: AdaptiveRadialCasimirResult) -> tuple[bool, str]:
    if not bool(result.all_microscopic_nodes_certified):
        reason = str(result.termination_reason)
        details = list(
            dict.fromkeys(
                str(record.get("reason", ""))
                for record in result.unresolved_points
                if isinstance(record, Mapping) and record.get("reason")
            )
        )
        if details:
            suffix = "; ".join(details[:3])
            if len(details) > 3:
                suffix += f"; and {len(details) - 3} more distinct reasons"
            if suffix not in reason:
                reason = f"{reason}: {suffix}"
        return False, reason
''',
)
replace_once(
    path,
    '''        radial_error = primary_radial + audit_radial
        offset_error = np.abs(audit_values - primary_values)
        offset_tolerance = np.maximum(
''',
    '''        radial_error = primary_radial + audit_radial
        offset_error = np.abs(audit_values - primary_values)
        estimated_offset_error = offset_error + audit_radial
        offset_tolerance = np.maximum(
''',
)
replace_once(
    path,
    '''            "offset_differences_J_m2": offset_error.tolist(),
            "offset_tolerances_J_m2": offset_tolerance.tolist(),
''',
    '''            "offset_differences_J_m2": offset_error.tolist(),
            "estimated_offset_error_bounds_J_m2": estimated_offset_error.tolist(),
            "offset_tolerances_J_m2": offset_tolerance.tolist(),
''',
)
replace_once(
    path,
    '''                "estimated_offset_errors_J_m2": offset["offset_differences_J_m2"],
''',
    '''                "estimated_offset_errors_J_m2": offset[
                    "estimated_offset_error_bounds_J_m2"
                ],
''',
)
replace_once(
    path,
    '''            batch = evaluate(combined)
            if not batch.all_established:
                raise FixedCasimirExecutionError(
                    "prefetched comparison contains unresolved microscopic points"
                )
''',
    '''            # Persist unresolved records too; the radial run will propagate their
            # exact point-level reasons through its normal fail-closed result.
            evaluate(combined)
''',
)
replace_once(
    path,
    '''                    (
                        current_order,
                        config.primary_offset_fraction,
                        radial_round_cap,
                    ),
                )
            )
''',
    '''                    (
                        current_order,
                        config.primary_offset_fraction,
                        radial_round_cap,
                    ),
                    (
                        current_order,
                        config.audit_offset_fraction,
                        radial_round_cap,
                    ),
                )
            )
''',
)
