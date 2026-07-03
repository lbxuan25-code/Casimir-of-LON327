import numpy as np

from lno327.collective.ward import ward_metadata
from lno327.finite_q_primitives import ward_metadata as old_ward_metadata


def test_ward_metadata_matches_legacy_helper_and_is_pure_diagnostic():
    response = np.array(
        [[1.0, 0.2j, 0.1], [-0.2j, 0.7, 0.05j], [0.1, -0.05j, 0.4]],
        dtype=complex,
    )
    before = response.copy()
    q = np.array([0.1, -0.03])

    actual = ward_metadata(response, 0.01, q)
    expected = old_ward_metadata(response, 0.01, q)

    assert actual == expected
    assert set(actual) == {"left_norm", "right_norm", "max_norm"}
    np.testing.assert_allclose(response, before)
