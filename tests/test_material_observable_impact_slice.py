from __future__ import annotations

from pathlib import Path

import pytest

from lno327.casimir.material_geometry_qualification_campaign import (
    build_todo4_qualification_campaign,
    load_todo4_qualification_manifest,
)
from lno327.casimir.material_observable_impact_slice import (
    normalize_direct_plan_ids,
)


MANIFEST = Path("validation/configs/casimir/todo4_representative_v1.json")


def _campaign():
    return build_todo4_qualification_campaign(
        load_todo4_qualification_manifest(MANIFEST)
    )


def test_default_plan_selection_keeps_every_direct_plan_for_pairing():
    selected = normalize_direct_plan_ids(
        _campaign(),
        pairing_name="dwave",
        plan_ids=None,
    )
    assert selected == (
        "direct/axis_parallel/dwave",
        "direct/oblique_rotated/dwave",
    )


def test_explicit_plan_selection_can_isolate_oblique_dwave_case():
    selected = normalize_direct_plan_ids(
        _campaign(),
        pairing_name="dwave",
        plan_ids=("direct/oblique_rotated/dwave",),
    )
    assert selected == ("direct/oblique_rotated/dwave",)


@pytest.mark.parametrize(
    "plan_ids",
    [
        (),
        ("direct/oblique_rotated/dwave", "direct/oblique_rotated/dwave"),
        ("direct/axis_parallel/spm",),
        ("direct/missing/dwave",),
    ],
)
def test_invalid_explicit_plan_selections_fail_closed(plan_ids):
    with pytest.raises(ValueError):
        normalize_direct_plan_ids(
            _campaign(),
            pairing_name="dwave",
            plan_ids=plan_ids,
        )
