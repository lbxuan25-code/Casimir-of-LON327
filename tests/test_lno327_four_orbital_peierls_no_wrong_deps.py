from pathlib import Path


def test_four_orbital_peierls_has_no_wrong_dependencies():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/lno327/models/lno327_four_orbital/peierls.py").read_text()
    forbidden = (
        "finite_q_primitives",
        "finite_q_engine",
        "ward_response",
        "physical_ward_residuals",
        "response_interface",
        "reflection_input",
        "casimir",
        "scripts",
        "validation",
        "tb_fourier",
    )

    for needle in forbidden:
        assert needle not in text, f"{needle!r} found in four-orbital peierls.py"
