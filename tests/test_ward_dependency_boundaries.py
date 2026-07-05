from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_ward_response_modules_do_not_import_old_or_forbidden_layers():
    checks = {
        "src/lno327/response/normal_density_current.py": (
            "from lno327." + "conductivity",
            "from ." + "conductivity",
            "from lno327." + "tb_fourier",
            "from ." + "tb_fourier",
            "from lno327.models.lno327_four_orbital.normal",
            "from lno327.models.lno327_four_orbital.vertices",
            "ward_response",
            "ward_validation",
            "reflection_input",
            "casimir",
            "validation/",
            "scripts/",
        ),
        "src/lno327/collective/ward.py": (
            "ward_response",
            "ward_validation",
            "finite_q_engine",
            "reflection_input",
            "casimir",
            "validation/",
            "scripts/",
        ),
        "src/lno327/collective/validation.py": (
            "ward_response",
            "ward_validation",
            "finite_q_engine",
            "reflection_input",
            "casimir",
            "validation/",
            "scripts/",
        ),
    }
    for relative, forbidden in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text


def test_active_entrypoints_do_not_import_old_ward_modules():
    finite_q_engine = (ROOT / "src/lno327/workflows/finite_q_engine.py").read_text(encoding="utf-8")
    api = (ROOT / "src/lno327/api.py").read_text(encoding="utf-8")

    assert "from ." + "ward_response" not in finite_q_engine
    assert "from lno327." + "ward_response" not in finite_q_engine
    assert "from ." + "ward_validation" not in api
    assert "from lno327." + "ward_validation" not in api
    assert "from ." + "ward_response" not in api
    assert "from lno327." + "ward_response" not in api
