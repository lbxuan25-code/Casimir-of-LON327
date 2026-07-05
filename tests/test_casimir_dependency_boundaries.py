from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CASIMIR_PACKAGE_FILES = (
    ROOT / "src" / "lno327" / "casimir" / "__init__.py",
    ROOT / "src" / "lno327" / "casimir" / "setup.py",
    ROOT / "src" / "lno327" / "casimir" / "reflection.py",
    ROOT / "src" / "lno327" / "casimir" / "lifshitz.py",
    ROOT / "src" / "lno327" / "casimir" / "torque.py",
)


def test_casimir_package_does_not_import_old_response_or_conductivity_modules():
    forbidden = (
        "from lno327." + "conductivity",
        "from ." + "conductivity",
        "import lno327." + "conductivity",
        "response_conventions",
        "reflection_input",
        "ward_response",
        "ward_validation",
        "finite_q_engine",
        "validation/",
        "scripts/",
    )

    for path in CASIMIR_PACKAGE_FILES:
        source = path.read_text()
        for needle in forbidden:
            assert needle not in source, f"{path} should not contain {needle!r}"


def test_reflection_uses_electrodynamics_conductivity_boundary():
    source = (ROOT / "src" / "lno327" / "casimir" / "reflection.py").read_text()

    assert "lno327.electrodynamics.conductivity" in source


def test_public_api_does_not_fall_back_to_old_boundary_modules():
    source = (ROOT / "src" / "lno327" / "api.py").read_text()
    forbidden = (
        "from ." + "conductivity",
        "from lno327." + "conductivity",
        "from ." + "response_conventions",
        "from lno327." + "response_conventions",
        "from ." + "reflection_input",
        "from lno327." + "reflection_input",
        "from ." + "ward_validation",
        "from lno327." + "ward_validation",
    )

    for needle in forbidden:
        assert needle not in source
