from pathlib import Path


def test_finite_q_engine_uses_model_and_generic_primitives():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/lno327/workflows/finite_q_engine.py").read_text()
    core_text = (root / "src/lno327/response/finite_q_bdg.py").read_text()
    forbidden = (
        "from ." + "bdg_response import",
        "from ." + "conductivity import",
        "from .models.lno327_four_orbital.bdg import bdg_hamiltonian",
        "from .models.lno327_four_orbital.pairing import pairing_matrix",
        "bdg_finite_q_vector_vertex(",
        "bdg_finite_q_contact_vertex(",
        "thermal_expectation_bdg(",
        "density_vertex()",
        "np.linalg.eigh(",
    )

    for needle in forbidden:
        assert needle not in text, f"{needle!r} found in finite_q_engine.py"

    assert "finite_q_primitives" not in text
    assert "add_bubble" not in text
    assert "from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz" in text
    assert "spec.peierls_hamiltonian_vector_vertex" in core_text
    assert "spec.peierls_hamiltonian_contact_vertex" in core_text
    assert "finite_q_bdg_response_from_model_ansatz" in text
