import numpy as np

from lno327.collective.schur import apply_amplitude_phase_schur, apply_phase_only_schur


def test_phase_only_schur_minus_and_plus_match_matrix_formula():
    bare = np.array([[2.0, 0.1], [0.2, 3.0]], dtype=complex)
    left = np.array([1.0, 0.5], dtype=complex)
    right = np.array([0.25, 0.75], dtype=complex)
    kernel = 4.0 + 0.2j

    minus = apply_phase_only_schur(bare, left, kernel, right, sign="minus")
    plus = apply_phase_only_schur(bare, left, kernel, right, sign="plus")

    np.testing.assert_allclose(minus.corrected_response, bare - np.outer(left, right) / kernel)
    np.testing.assert_allclose(plus.corrected_response, bare + np.outer(left, right) / kernel)
    assert minus.status == "minus_phase_schur_applied"
    assert plus.status == "plus_phase_schur_applied"


def test_phase_only_schur_zero_kernel_skips_correction():
    bare = np.eye(2, dtype=complex)
    result = apply_phase_only_schur(bare, np.ones(2), 0.0, np.ones(2), sign="minus")
    np.testing.assert_allclose(result.corrected_response, bare)
    assert result.status == "skipped_zero_phase_kernel"
    assert result.inverse_method == "not_used"


def test_amplitude_phase_schur_inv_and_pinv_branches():
    bare = np.array([[2.0, 0.1], [0.2, 3.0]], dtype=complex)
    left = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=complex)
    kernel = np.array([[4.0, 0.2], [0.1, 2.0]], dtype=complex)
    right = np.array([[0.5, 0.1], [0.3, 0.8]], dtype=complex)

    regular = apply_amplitude_phase_schur(bare, left, kernel, right)
    np.testing.assert_allclose(regular.corrected_response, bare - left @ np.linalg.inv(kernel) @ right)
    assert regular.inverse_method == "inv"

    singular = apply_amplitude_phase_schur(bare, left, np.diag([1.0, 1e-16]), right, condition_threshold=1e6)
    np.testing.assert_allclose(
        singular.corrected_response,
        bare - left @ np.linalg.pinv(np.diag([1.0, 1e-16])) @ right,
    )
    assert singular.inverse_method == "pinv_diagnostic"
    assert singular.status == "applied_with_pinv_diagnostic"
