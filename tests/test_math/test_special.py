# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test special functions of the math backend"""

import numpy as np
from scipy.special import eval_hermite, factorial

from mrmustard import math


def test_reduction_to_renorm_physicists_polys():
    """Tests that the math interface provides the expected renormalized Hermite polys"""
    x = np.arange(-1, 1, 0.1)
    init = 1
    n_max = 5
    A = -np.ones([init, init], dtype=complex)
    vals = np.array(
        [
            math.hermite_renormalized(2 * A, 2 * np.array([x0], dtype=complex), 1, (n_max,))
            for x0 in x
        ],
    ).T
    expected = np.array([eval_hermite(i, x) / np.sqrt(factorial(i)) for i in range(len(vals))])
    assert np.allclose(vals, expected)
