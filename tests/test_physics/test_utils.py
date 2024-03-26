# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the helper functions in physics."""

import numpy as np
from mrmustard import math
from mrmustard.physics.utils import real_gaussian_integral


def test_real_gaussian_integral():
    """Tests the ``real_gaussian_integral`` method with a hard-coded A matric from a Gaussian(3) state."""
    A = math.astensor(
        np.array(
            [
                [0.35307718 - 0.09738001j, -0.01297994 + 0.26050244j, 0.05349344 - 0.13728068j],
                [-0.01297994 + 0.26050244j, 0.05696707 - 0.2351408j, 0.18954838 - 0.42959383j],
                [0.05349344 - 0.13728068j, 0.18954838 - 0.42959383j, -0.16931712 - 0.09205837j],
            ]
        )
    )
    b = math.astensor(np.arange(3) + 0j)
    c = 1.0 + 0j
    res = real_gaussian_integral((A, b, c), idx=[0, 1])
    assert np.allclose(res[0], A[2, 2] - A[2:, :2] @ math.inv(A[:2, :2]) @ A[:2, 2:])
    assert np.allclose(res[1], b[2] - math.sum(A[2:, :2] * math.matvec(math.inv(A[:2, :2]), b[:2])))
    assert np.allclose(
        res[2],
        c
        * (2 * np.pi)
        / math.sqrt(math.det(A[:2, :2]))
        * math.exp(-0.5 * math.sum(b[:2] * math.matvec(math.inv(A[:2, :2]), b[:2]))),
    )
