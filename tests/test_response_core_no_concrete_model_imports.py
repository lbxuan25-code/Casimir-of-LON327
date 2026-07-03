from pathlib import Path


CORE_FILES = (
    "src/lno327/bdg/kinematics.py",
    "src/lno327/bdg/nambu.py",
    "src/lno327/bdg/spectrum.py",
    "src/lno327/response/containers.py",
    "src/lno327/response/occupations.py",
    "src/lno327/response/bubble.py",
)


def test_response_core_has_no_concrete_model_imports_or_fixed_bdg_shapes():
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "lno327_four_orbital",
        "symmetry_bdg_2band",
        "PairingAmplitudes",
        "PairingKind",
        "NormalStateParameters",
        "np.zeros((4, 4)",
        "np.zeros((8, 8)",
    )

    for relative_path in CORE_FILES:
        text = (root / relative_path).read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {relative_path}"
