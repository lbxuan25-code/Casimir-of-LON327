"""Plan-filtered observable-impact diagnostics for economical high-N follow-up.

This wrapper preserves the frozen full campaign plan while allowing one or more
explicit direct-plan IDs to be selected for a diagnostic observable replay. It
never performs microscopic integration, reads or writes the certified response
cache, or promotes unresolved responses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lno327.casimir.material_geometry_qualification_campaign import (
    Todo4QualificationCampaign,
)
from lno327.casimir.material_geometry_qualification_execution import require_frozen_plan
from lno327.casimir.material_geometry_qualification_io import (
    atomic_write_json,
    slug,
    source_commit,
)
from lno327.casimir.material_observable_impact_calibration import (
    build_observable_impact_calibration,
    write_observable_impact_calibration,
)


def normalize_direct_plan_ids(
    campaign: Todo4QualificationCampaign,
    *,
    pairing_name: str,
    plan_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate an explicit subset of direct plans for one pairing."""
    if not isinstance(campaign, Todo4QualificationCampaign):
        raise TypeError("campaign must be a Todo4QualificationCampaign")
    pairing = str(pairing_name)
    available = tuple(
        entry.plan_id
        for entry in campaign.entries
        if entry.kind == "direct" and entry.pairing_name == pairing
    )
    if not available:
        raise ValueError(f"campaign has no direct entries for pairing {pairing!r}")
    if plan_ids is None:
        return available

    selected = tuple(str(value) for value in plan_ids)
    if not selected:
        raise ValueError("plan_ids must be nonempty when supplied")
    if len(set(selected)) != len(selected):
        raise ValueError("plan_ids must be unique")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(
            "requested plan_ids are not direct plans for pairing "
            f"{pairing!r}: {unknown}"
        )
    return selected


def write_plan_filtered_observable_impact_calibration(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    diagnostic_source_dir: Path,
    pairing_name: str = "dwave",
    plan_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write observable-impact evidence for an explicit direct-plan subset."""
    selected = normalize_direct_plan_ids(
        campaign,
        pairing_name=pairing_name,
        plan_ids=plan_ids,
    )
    if plan_ids is None:
        return write_observable_impact_calibration(
            campaign,
            output_dir=output_dir,
            diagnostic_source_dir=diagnostic_source_dir,
            pairing_name=pairing_name,
        )

    frozen = require_frozen_plan(campaign, output_dir)
    selected_set = set(selected)
    selected_entries = tuple(
        entry
        for entry in campaign.entries
        if entry.plan_id in selected_set
    )
    if tuple(entry.plan_id for entry in selected_entries) != selected:
        by_id = {entry.plan_id: entry for entry in selected_entries}
        selected_entries = tuple(by_id[plan_id] for plan_id in selected)

    sliced_campaign = Todo4QualificationCampaign(
        manifest=campaign.manifest,
        manifest_sha256=campaign.manifest_sha256,
        entries=selected_entries,
    )
    payload = build_observable_impact_calibration(
        sliced_campaign,
        diagnostic_source_dir=diagnostic_source_dir,
        pairing_name=pairing_name,
    )
    payload["source_commit"] = source_commit()
    payload["current_plan_sha256"] = frozen["plan_sha256"]
    payload["selected_plan_ids"] = list(selected)
    payload["full_campaign_plan_preserved"] = True
    payload["plan_filter_applied"] = True

    selection_tag = "__".join(slug(plan_id) for plan_id in selected)
    destination = (
        Path(output_dir)
        / "observable_impact"
        / str(payload["source_diagnostics"]["diagnostic_ladder_tag"])
        / f"{pairing_name}__{selection_tag}.json"
    )
    atomic_write_json(destination, payload)
    return payload


__all__ = [
    "normalize_direct_plan_ids",
    "write_plan_filtered_observable_impact_calibration",
]
