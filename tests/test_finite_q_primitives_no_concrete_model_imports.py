from pathlib import Path


def test_new_finite_q_primitive_sources_have_no_concrete_model_imports():
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "src/lno327/bdg/finite_q.py",
        "src/lno327/response/finite_q.py",
    )
    forbidden = (
        "lno327_four_orbital",
        "symmetry_bdg_2band",
        "PairingAmplitudes",
        "PairingKind",
        "NormalStateParameters",
        "pairing_matrix",
        "tb_fourier",
        "peierls_hamiltonian",
        "ward_response",
        "physical_ward_residuals",
        "from lno327.models",
        "from .models",
        "np.zeros((4, 4)",
        "np.zeros((8, 8)",
        "np.eye(4",
        "np.eye(8",
    )

    for relative_path in source_files:
        text = (root / relative_path).read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {relative_path}"
