from __future__ import annotations

import pytest

from lno327.casimir.material_geometry_qualification_diagnostics import (
    diagnostic_ladder_tag,
    normalize_diagnostic_n_candidates,
)


def test_base_diagnostic_ladder_is_preserved_without_override() -> None:
    base = (128, 192, 256)

    assert normalize_diagnostic_n_candidates(base, None) == base
    assert diagnostic_ladder_tag(base) == "N128-192-256"


def test_extended_diagnostic_ladder_requires_exact_overlap_anchor() -> None:
    base = (128, 192, 256)

    assert normalize_diagnostic_n_candidates(base, (256, 384, 512)) == (
        256,
        384,
        512,
    )

    with pytest.raises(ValueError, match="final base N overlap anchor"):
        normalize_diagnostic_n_candidates(base, (384, 512, 768))


def test_extended_diagnostic_ladder_rejects_unsafe_shapes() -> None:
    base = (128, 192, 256)

    with pytest.raises(ValueError, match="at least three"):
        normalize_diagnostic_n_candidates(base, (256, 384))
    with pytest.raises(ValueError, match="positive even"):
        normalize_diagnostic_n_candidates(base, (256, 383, 512))
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_diagnostic_n_candidates(base, (256, 512, 384))
