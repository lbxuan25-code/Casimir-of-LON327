from pathlib import Path


def test_peierls_model_layer_sources_have_no_wrong_dependencies():
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "src/lno327/models/hopping.py",
        "src/lno327/models/lno327_four_orbital/peierls.py",
        "src/lno327/models/symmetry_bdg_2band/peierls.py",
    )
    generic_forbidden = (
        "lno327_four_orbital",
        "symmetry_bdg_2band",
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
    model_forbidden = generic_forbidden[2:]

    for relative_path in source_files:
        text = (root / relative_path).read_text()
        forbidden = generic_forbidden if relative_path == "src/lno327/models/hopping.py" else model_forbidden
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {relative_path}"
