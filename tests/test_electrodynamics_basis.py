from __future__ import annotations

import numpy as np
import pytest

from lno327.electrodynamics.basis import (
    q_crystal_to_lab,
    q_lab_to_crystal,
    rotation_matrix,
    tensor_crystal_to_lab,
    tensor_crystal_to_lab_lt,
    tensor_lab_to_crystal,
    tensor_lt_to_xy,
    tensor_xy_to_lt,
    xy_to_lt_rotation,
)
from lno327.electrodynamics.reflection import rotate_sigma_tilde_xy_to_lt


def test_rotation_matrix_is_orthogonal_and_uses_active_ccw_convention():
    rotation = rotation_matrix(np.pi / 2.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0]), np.array([0.0, 1.0]), atol=1e-15)


def test_q_crystal_lab_round_trip():
    q_lab = np.array([0.21, -0.13])
    theta = 0.37
    q_crystal = q_lab_to_crystal(q_lab, theta)
    np.testing.assert_allclose(q_crystal_to_lab(q_crystal, theta), q_lab, rtol=1e-14, atol=1e-14)


def test_rank_two_tensor_crystal_lab_round_trip():
    tensor_crystal = np.array([[3.0, 0.4], [0.2, 1.0]], dtype=complex)
    theta = -0.42
    tensor_lab = tensor_crystal_to_lab(tensor_crystal, theta)
    np.testing.assert_allclose(
        tensor_lab_to_crystal(tensor_lab, theta),
        tensor_crystal,
        rtol=1e-14,
        atol=1e-14,
    )


def test_xy_lt_transform_uses_L_parallel_q_and_T_equal_z_cross_L():
    projection = xy_to_lt_rotation(0.0, 2.0)
    np.testing.assert_allclose(projection, np.array([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-15)

    tensor_xy = np.array([[2.0, 0.3], [0.3, 5.0]], dtype=complex)
    tensor_lt = tensor_xy_to_lt(tensor_xy, 0.0, 2.0)
    np.testing.assert_allclose(tensor_lt, np.array([[5.0, -0.3], [-0.3, 2.0]]), atol=1e-15)
    np.testing.assert_allclose(tensor_lt_to_xy(tensor_lt, 0.0, 2.0), tensor_xy, atol=1e-15)


def test_direct_crystal_to_lab_lt_matches_explicit_two_step_transform():
    q_lab = np.array([0.17, 0.09])
    theta = 0.61
    tensor_crystal = np.array([[4.0, 0.7], [0.7, 1.5]], dtype=complex)

    direct = tensor_crystal_to_lab_lt(tensor_crystal, q_lab, theta)
    tensor_lab = tensor_crystal_to_lab(tensor_crystal, theta)
    explicit = tensor_xy_to_lt(tensor_lab, q_lab[0], q_lab[1])
    np.testing.assert_allclose(direct, explicit, rtol=1e-14, atol=1e-14)


def test_reflection_compatibility_wrapper_uses_central_transform():
    tensor_xy = np.array([[2.0, 0.1j], [-0.1j, 3.0]], dtype=complex)
    expected = tensor_xy_to_lt(tensor_xy, 0.3, -0.4)
    actual = rotate_sigma_tilde_xy_to_lt(tensor_xy, 0.3, -0.4)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_lt_basis_rejects_q_zero_unless_explicitly_allowed():
    with pytest.raises(ValueError, match="nonzero"):
        xy_to_lt_rotation(0.0, 0.0)
    np.testing.assert_array_equal(xy_to_lt_rotation(0.0, 0.0, allow_q_zero=True), np.eye(2))
