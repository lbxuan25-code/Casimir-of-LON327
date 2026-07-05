from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_finite_q_engine_is_public_lno327_adapter_only():
    text = (ROOT / "src/lno327/finite_q_engine.py").read_text(encoding="utf-8")
    forbidden = (
        "from ." + "finite_q_primitives import",
        "from ." + "bdg_response import",
        "from ." + "conductivity import",
        "add_bubble",
        "spec.peierls_hamiltonian_vector_vertex",
        "spec.peierls_hamiltonian_contact_vertex",
        "np.linalg.eigh(",
    )
    for needle in forbidden:
        assert needle not in text
    assert "from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz" in text
    assert "from lno327.collective.schur import" in text


def test_model_driven_core_has_no_concrete_four_orbital_imports_or_legacy_primitives():
    text = (ROOT / "src/lno327/response/finite_q_bdg.py").read_text(encoding="utf-8")
    forbidden = (
        "lno327." + "bdg_response",
        "lno327." + "finite_q_primitives",
        "lno327." + "conductivity",
        "lno327.models.lno327_four_orbital.bdg",
        "lno327.models.lno327_four_orbital.pairing",
        "lno327." + "reflection_input",
        "lno327.casimir",
        "scripts",
    )
    for needle in forbidden:
        assert needle not in text
    assert "from lno327.response.validation import validate_finite_q_inputs" in text


def test_collective_helpers_do_not_import_engine_or_casimir_layers():
    for relative in ("src/lno327/collective/schur.py", "src/lno327/collective/ward.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "finite_q_engine" not in text
        assert "reflection_input" not in text
        assert "casimir" not in text
