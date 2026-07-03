from pathlib import Path


def test_nonlocal_bdg_source_has_no_concrete_model_imports_or_fixed_bdg_shapes():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/lno327/response/nonlocal_bdg.py").read_text()
    forbidden = (
        "lno327_four_orbital",
        "symmetry_bdg_2band",
        "PairingAmplitudes",
        "PairingKind",
        "NormalStateParameters",
        "pairing_matrix",
        "from lno327.models",
        "from .models",
        "np.zeros((4, 4)",
        "np.zeros((8, 8)",
    )

    for needle in forbidden:
        assert needle not in text, f"{needle!r} found in response/nonlocal_bdg.py"
