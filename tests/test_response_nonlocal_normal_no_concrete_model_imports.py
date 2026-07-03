from pathlib import Path


def test_nonlocal_normal_sources_have_no_concrete_model_imports_or_fixed_bdg_shapes():
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "src/lno327/response/nonlocal_normal.py",
        "src/lno327/response/validation.py",
        "src/lno327/response/bubble.py",
    )
    forbidden = (
        "lno327_four_orbital",
        "symmetry_bdg_2band",
        "PairingAmplitudes",
        "PairingKind",
        "NormalStateParameters",
        "np.zeros((4, 4)",
        "np.zeros((8, 8)",
    )

    for relative_path in source_files:
        text = (root / relative_path).read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {relative_path}"
