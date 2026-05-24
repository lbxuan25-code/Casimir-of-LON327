from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_bdg_paramagnetic_kernel.py"
SPEC = spec_from_file_location("diagnose_bdg_paramagnetic_kernel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnose = module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def test_spm_and_dwave_scans_return_nonempty_data():
    for kind in ("spm", "dwave"):
        data = diagnose.scan_kind(
            kind=kind,
            delta0_eV=0.04,
            nk=4,
            temperature_K=30.0,
            matsubara_min=1,
            matsubara_max=2,
            eta_eV=1e-4,
        )

        assert data["omega_eV"].size > 0
        assert data["Kxx"].size > 0
        assert np.isfinite(data["Kxx"]).all()
        assert np.isfinite(data["Kyy"]).all()


def test_scan_length_matches_matsubara_point_count():
    data = diagnose.scan_kind(
        kind="spm",
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_min=2,
        matsubara_max=5,
        eta_eV=1e-4,
    )

    assert data["n"].size == 4
    assert data["omega_eV"].size == 4
    assert data["relative_offdiag"].size == 4


def test_spm_and_dwave_are_c4_symmetric_without_magnetic_field():
    for kind in ("spm", "dwave"):
        data = diagnose.scan_kind(
            kind=kind,
            delta0_eV=0.04,
            nk=8,
            temperature_K=30.0,
            matsubara_min=1,
            matsubara_max=2,
            eta_eV=1e-4,
        )

        assert np.max(np.abs(data["delta_K"])) < 1e-10
        assert np.max(data["relative_offdiag"]) < 1e-10


def test_saved_npz_fields_are_complete(tmp_path):
    data = diagnose.scan_kind(
        kind="spm",
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_min=1,
        matsubara_max=1,
        eta_eV=1e-4,
    )
    npz_path, _, _ = diagnose.save_outputs(data, tmp_path / "kernel_scan")

    with np.load(npz_path) as loaded:
        assert set(loaded.files) == diagnose.REQUIRED_NPZ_FIELDS
