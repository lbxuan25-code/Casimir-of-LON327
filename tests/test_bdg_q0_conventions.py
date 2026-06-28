from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np

import lno327
from lno327.bdg_q0_conventions import evaluate_bdg_q0_convention
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh
from lno327.pairing import PairingAmplitudes

ROOT = Path(__file__).resolve().parents[1]


def _inputs(nk: int = 3):
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = PairingAmplitudes(delta0_eV=0.04)
    return points, weights, config, amp


def test_q0_helper_decomposes_local_k_para_and_reports_expected_statuses():
    points, weights, config, amp = _inputs()
    spm = evaluate_bdg_q0_convention("spm", points, weights, config, amp)
    dwave = evaluate_bdg_q0_convention("dwave", points, weights, config, amp)

    assert spm.status == "convention_aware_pass"
    assert dwave.status == "intraband_aware_pass"
    for result in (spm, dwave):
        np.testing.assert_allclose(
            result.local_k_para_total,
            result.local_k_para_interband + result.local_k_para_intraband,
            rtol=1e-9,
            atol=1e-10,
        )
        assert result.current_vertex_status == "vertex_operator_q0_match"
        assert result.valid_for_casimir_input is False


def test_dwave_raw_vs_total_mismatch_is_visible_but_explained_by_intraband():
    points, weights, config, amp = _inputs()
    dwave = evaluate_bdg_q0_convention("dwave", points, weights, config, amp)
    comparisons = dwave.comparison_by_name
    assert not comparisons["raw_vs_total"].passes_tolerance
    assert comparisons["raw_vs_interband"].passes_tolerance
    assert comparisons["total_minus_raw_vs_intraband"].passes_tolerance
    assert "intraband" in dwave.interpretation


def test_bdg_finite_q_status_contract_records_expected_states_and_forbidden_stale_terms():
    text = (ROOT / "validation" / "contracts" / "bdg_finite_q_status.yaml").read_text(encoding="utf-8")
    assert "spm: convention_aware_pass" in text
    assert "dwave: intraband_aware_pass" in text
    assert "ward_identity_closed: false" in text
    assert "valid_for_casimir_input: false" in text
    assert "dwave_specific_raw_bubble_mismatch" in text
    assert "formal_casimir_input" in text


def test_public_package_shape_has_single_lno327_entrypoint():
    from lno327.api import KuboConfig as ApiKuboConfig
    from lno327.api import PairingAmplitudes as ApiPairingAmplitudes
    from lno327.api import local_response_imag_axis

    assert lno327.__name__ == "lno327"
    assert ApiKuboConfig is KuboConfig
    assert ApiPairingAmplitudes is PairingAmplitudes
    assert callable(local_response_imag_axis)
    assert importlib.util.find_spec("lno_327") is None


def test_repository_root_python_smoke_imports_public_api():
    code = (
        "import lno327; "
        "from lno327.api import KuboConfig, PairingAmplitudes, local_response_imag_axis; "
        "print('lno327 import smoke ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "lno327 import smoke ok"
