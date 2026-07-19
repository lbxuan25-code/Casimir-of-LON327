from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "src/lno327/casimir/certified_point_provider.py"
replace_once(
    path,
    '''def certified_primary_logdet(point: Mapping[str, Any]) -> float:
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
''',
    '''def certified_primary_logdet(point: Mapping[str, Any]) -> float:
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
    if not bool(row.get("hard_physical_closure_across_shifts")):
        raise ValueError("point audit row lacks hard-physical closure across shifts")
    shifts = row.get("shifts", {})
    if not isinstance(shifts, Mapping) or not shifts:
        raise ValueError("audit_N history row has no shift states")
    primary_labels = [
        label for label in shifts if str(label).startswith("shift_0:")
    ]
    if len(primary_labels) != 1:
        raise ValueError("audit_N history row does not identify exactly one primary shift")
    primary = shifts[primary_labels[0]]
    value = float(primary["two_plate_logdet"])
    if not np.isfinite(value) or not bool(primary.get("hard_physical_passed")):
        raise ValueError("primary audit shift is not a finite hard-physical point")
    cross_shift = row.get("two_plate_logdet_cross_shift", {})
    if not isinstance(cross_shift, Mapping) or not bool(cross_shift.get("passed")):
        raise ValueError("point audit cross-shift comparison did not pass")
    return value
''',
)
replace_once(
    path,
    '''                "status": "succeeded",
                "requested_q_count": int(requested_q_count),
                "requested_point_count": int(requested_point_count),
                "matsubara_indices": [
''',
    '''                "status": "succeeded",
                "requested_q_count": int(requested_q_count),
                "requested_point_count": int(requested_point_count),
                "returned_point_count": int(len(payload.get("point_results", ()))),
                "unresolved_point_count": int(
                    sum(
                        1
                        for point in payload.get("point_results", ())
                        if isinstance(point, Mapping)
                        and point.get("sweet_spot", {}).get("status") != "established"
                    )
                ),
                "matsubara_indices": [
''',
)
replace_once(
    path,
    '''        new_point_count = 0
        new_batch_count = 0
        for missing_indices, group in groups.items():
''',
    '''        new_point_count = 0
        new_batch_count = 0
        cache_dirty = False
        for missing_indices, group in groups.items():
''',
)
replace_once(
    path,
    '''                    self._consume_payload(
                        certification.payload,
                        labels=labels,
                        q_keys=tuple(q_key for q_key, _ in chunk),
                        requested_config=run_config,
                    )
                    self._consume_certifier_telemetry(
''',
    '''                    self._consume_payload(
                        certification.payload,
                        labels=labels,
                        q_keys=tuple(q_key for q_key, _ in chunk),
                        requested_config=run_config,
                    )
                    cache_dirty = True
                    self._consume_certifier_telemetry(
''',
)
replace_once(
    path,
    '''                if (
                    self.cache_path is not None
                    and len(group) > self.certifier_q_batch_size
                ):
                    self._save()

        cache_hit_point_count = requested_point_count - new_point_count
''',
    '''                if (
                    self.cache_path is not None
                    and len(group) > self.certifier_q_batch_size
                ):
                    self._save()
                    cache_dirty = False

        cache_hit_point_count = requested_point_count - new_point_count
''',
)
replace_once(path, '''        if groups:
            self._save()
''', '''        if cache_dirty:
            self._save()
''')
