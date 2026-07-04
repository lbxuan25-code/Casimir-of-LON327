import numpy as np

from lno327.collective.validation import validate_physical_ward_identity as new_validate
from lno327.ward_validation import validate_physical_ward_identity as old_validate


def _compare_reports(new, old):
    np.testing.assert_allclose(new.left_residual, old.left_residual)
    np.testing.assert_allclose(new.right_residual, old.right_residual)
    assert new.left_norm == old.left_norm
    assert new.right_norm == old.right_norm
    assert new.passed is old.passed
    assert new.tolerance == old.tolerance
    assert new.ward_vectors == old.ward_vectors
    assert new.notes == old.notes


def test_collective_validation_matches_old_reference_for_pass_and_fail_cases():
    q = np.array([0.1, -0.03])
    zero = np.zeros((3, 3), dtype=complex)
    response = np.array(
        [[1.0, 0.2j, 0.1], [-0.2j, 0.7, 0.05j], [0.1, -0.05j, 0.4]],
        dtype=complex,
    )
    for matrix, tolerance in ((zero, 1e-12), (response, 1e-12)):
        new = new_validate(matrix, 0.01, q, tolerance=tolerance, notes=("diagnostic only",))
        old = old_validate(matrix, 0.01, q, tolerance=tolerance, notes=("diagnostic only",))
        _compare_reports(new, old)


def test_collective_validation_copies_residuals_and_preserves_input():
    q = np.array([0.1, -0.03])
    response = np.eye(3, dtype=complex)
    before = response.copy()
    report = new_validate(response, 0.01, q)
    report.left_residual[0] = 999.0

    np.testing.assert_allclose(response, before)
    second = new_validate(response, 0.01, q)
    assert second.left_residual[0] != 999.0
