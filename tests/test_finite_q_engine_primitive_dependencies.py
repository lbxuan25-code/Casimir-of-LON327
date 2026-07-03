from pathlib import Path


def test_finite_q_engine_uses_model_and_generic_primitives():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/lno327/finite_q_engine.py").read_text()
    forbidden = (
        "from .bdg_response import",
        "from .conductivity import",
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

    assert "from .finite_q_primitives import ward_metadata" in text
    assert "add_bubble" not in text.split("from .finite_q_primitives import", maxsplit=1)[1].split("\n", maxsplit=1)[0]
    assert "spec.peierls_hamiltonian_vector_vertex" in text
    assert "spec.peierls_hamiltonian_contact_vertex" in text
    assert "finite_q_bdg_response_from_model_ansatz" in text
