from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_electrodynamics_new_modules_do_not_import_old_or_forbidden_layers():
    checks = {
        "src/lno327/electrodynamics/units.py": (
            "response_conventions",
            "reflection_input",
            "lno327.casimir",
            "from .casimir",
            "lno327." + "conductivity",
            "from ." + "conductivity",
            "ward_response",
            "ward_validation",
            "validation/",
            "scripts/",
        ),
        "src/lno327/electrodynamics/conventions.py": (
            "response_conventions",
            "reflection_input",
            "lno327.casimir",
            "from .casimir",
            "lno327." + "conductivity",
            "from ." + "conductivity",
            "ward_response",
            "ward_validation",
            "validation/",
            "scripts/",
        ),
        "src/lno327/electrodynamics/reflection.py": (
            "response_conventions",
            "reflection_input",
            "lno327.casimir",
            "from .casimir",
            "lno327." + "conductivity",
            "from ." + "conductivity",
            "ward_response",
            "ward_validation",
            "validation/",
            "scripts/",
        ),
    }
    for relative, forbidden in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text


def test_active_interfaces_do_not_import_top_level_response_conventions():
    local_interface = (ROOT / "src/lno327/response/local_interface.py").read_text(encoding="utf-8")
    api = (ROOT / "src/lno327/api.py").read_text(encoding="utf-8")

    assert "lno327." + "response_conventions" not in local_interface
    assert "from lno327." + "response_conventions" not in local_interface
    assert "from ." + "response_conventions" not in local_interface
    assert "from ." + "response_conventions" not in api
    assert "from lno327." + "response_conventions" not in api
