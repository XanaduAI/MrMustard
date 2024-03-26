import numpy as np

from mrmustard import math
from mrmustard.physics.utils import real_gaussian_integral


def test_real_gaussian_integral():
    """Tests the ``real_gaussian_integral`` method."""
    A0 = np.random.random((3, 3))
    A = (A0 + A0.T) / 2
    b = np.arange(3)
    c = 1.0
    res = real_gaussian_integral((A, b, c), idx=[0, 1])
    assert np.allclose(res[0], A[2, 2] - A[2:, :2] @ np.linalg.inv(A[:2, :2]) @ A[:2, 2:])
    assert np.allclose(res[1], b[2] - A[2:, :2] @ np.linalg.inv(A[:2, :2]) @ b[:2])
    assert np.allclose(
        res[2],
        c
        * (2 * np.pi)
        / np.sqrt(np.linalg.det(A[:2, :2]))
        * np.exp(-0.5 * b[:2] @ np.linalg.inv(A[:2, :2]) @ b[:2]),
    )
