# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the fast diagonal module"""


import numpy as np
from mrmustard import math
from mrmustard.lab_dev import DM, Dgate


def test_fast_diagonal_2modes():
    r"""Test that the fast diagonal function works for a 2-mode Gaussian state."""
    A, b, c = (DM.random([0, 1]) >> Dgate([0, 1], x=0.4, y=0.6)).bargmann_triple()
    fd = math.hermite_renormalized_1leftoverMode(A, b, c, (5, 10))
    control = math.hermite_renormalized(A, b, c, (5, 10, 5, 10))
    control = control[:, np.arange(10), :, np.arange(10)]  # shape (10,)+(5,5)
    assert np.allclose(fd, control.transpose(1, 2, 0))


def test_fast_diagonal_3modes():
    r"""Test that the fast diagonal function works for a 3-mode Gaussian state."""
    A, b, c = (DM.random([0, 1, 2]) >> Dgate([0, 1, 2], x=0.4, y=0.6)).bargmann_triple()
    fd = math.hermite_renormalized_1leftoverMode(A, b, c, (3, 4, 5))
    control = math.hermite_renormalized(A, b, c, (3, 4, 5, 3, 4, 5))
    control = control[:, :, np.arange(5), :, :, np.arange(5)]  # shape (5,)+(3,4,3,4)
    control = control[:, :, np.arange(4), :, np.arange(4)]  # shape (4,5)+(3,3)
    control = control.transpose(2, 3, 0, 1)
    assert np.allclose(fd, control)
