"""One-shot mechanical migration of the fixed-point certification engine."""
from __future__ import annotations

from pathlib import Path


SOURCE = Path("validation/lib/transverse_point_sweet_spot_engine_legacy.py")
OUTPUT = Path("src/lno327/casimir/fixed_point_engine.py")
TEST = Path("tests/test_casimir_fixed_point_engine_migration.py")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    replacements = {
        "from validation.lib.finite_q_validation_models import get_finite_q_validation_model": (
            "from lno327.casimir.microscopic_model import "
            "get_finite_q_microscopic_model"
        ),
        "from validation.lib.matsubara import matsubara_energy_eV": (
            "from lno327.casimir.matsubara import matsubara_energy_eV"
        ),
        "get_finite_q_validation_model(": "get_finite_q_microscopic_model(",
        "validation/outputs/matsubara/transverse_point_sweet_spot/diagnostic.json": (
            "outputs/casimir/current/fixed_point_certification.json"
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise SystemExit(f"required migration token missing: {old}")
        text = text.replace(old, new)

    if "from validation" in text or "import validation" in text:
        raise SystemExit("production engine still imports validation")

    OUTPUT.write_text(text, encoding="utf-8")
    SOURCE.write_text(
        '''"""Compatibility facade for the production fixed-point engine.

The complete numerical implementation now lives in
:mod:`lno327.casimir.fixed_point_engine`. Validation retains this module path only
so historical diagnostics and tests continue to call the same implementation.
"""
from __future__ import annotations

from lno327.casimir import fixed_point_engine as _production

for _name in dir(_production):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_production, _name))
''',
        encoding="utf-8",
    )
    TEST.write_text(
        '''"""Identity guards for the mechanically migrated fixed-point engine."""
from __future__ import annotations


def test_validation_legacy_engine_is_production_engine_surface() -> None:
    from lno327.casimir import fixed_point_engine as production
    from validation.lib import transverse_point_sweet_spot_engine_legacy as facade

    for name in (
        "assess_frequency_level",
        "_build_context_jobs",
        "_execute_level",
        "_plate_state",
        "_two_plate_state",
        "main",
    ):
        assert getattr(facade, name) is getattr(production, name)


def test_cpu_headroom_facade_still_wraps_production_parser() -> None:
    from lno327.casimir import fixed_point_engine as production
    from validation.lib import transverse_point_sweet_spot_engine as public_facade

    assert public_facade.assess_frequency_level is production.assess_frequency_level
    assert public_facade._legacy._parse_args is production._parse_args
''',
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
