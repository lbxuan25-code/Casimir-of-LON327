import pytest

from lno327.bdg.kinematics import MomentumTransfer, shifted_momenta


def test_q_zero_left_and_right_equal_k():
    kx, ky = 0.37, -0.41

    left, right = shifted_momenta(kx, ky)

    assert left == (kx, ky)
    assert right == (kx, ky)


def test_finite_q_uses_symmetric_half_shift():
    kx, ky = 0.37, -0.41
    qx, qy = 0.2, -0.6

    left, right = shifted_momenta(kx, ky, qx, qy)

    assert left == (kx + 0.5 * qx, ky + 0.5 * qy)
    assert right == (kx - 0.5 * qx, ky - 0.5 * qy)


def test_invalid_convention_raises_value_error():
    transfer = MomentumTransfer(qx=0.1, qy=0.2, convention="left_shift")

    with pytest.raises(ValueError, match="symmetric_q_over_2"):
        transfer.left(0.3, 0.4)
