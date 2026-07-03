from pathlib import Path


def test_migrated_response_sources_have_no_concrete_model_imports_or_fixed_bdg_shapes():
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "src/lno327/response/config.py",
        "src/lno327/response/local_normal.py",
        "src/lno327/response/local_bdg.py",
        "src/lno327/electrodynamics/conductivity.py",
        "src/lno327/numerics/matsubara.py",
        "src/lno327/numerics/grids.py",
        "src/lno327/numerics/weights.py",
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
        path = root / relative_path
        if not path.exists():
            continue
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} found in {relative_path}"
