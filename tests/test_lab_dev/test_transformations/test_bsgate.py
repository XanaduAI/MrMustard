# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``BSgate`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import BSgate


class TestBSgate:
    r"""
    Tests for the ``BSgate`` class.
    """

    modes = [[0, 8], [1, 2], [9, 7]]
    theta = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    def test_init(self):
        gate = BSgate([0, 1], 2, 3)

        assert gate.name == "BSgate"
        assert gate.modes == [0, 1]
        assert gate.theta.value == 2
        assert gate.phi.value == 3

    def test_init_error(self):
        with pytest.raises(ValueError, match="Expected a pair"):
            BSgate([1, 2, 3])

    def test_representation(self):
        rep1 = BSgate([0, 1], 0.1, 0.2).ansatz
        A_exp = [
            [
                [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
                [0.0, 0, 0.0978434 + 0.01983384j, 0.99500417],
                [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
                [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
            ]
        ]
        assert math.allclose(rep1.A, A_exp)
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [1])

        rep2 = BSgate([0, 1], 0.1).ansatz
        A_exp = [
            [
                [0, 0, 9.95004165e-01, -9.98334166e-02],
                [0.0, 0, 9.98334166e-02, 9.95004165e-01],
                [9.95004165e-01, 9.98334166e-02, 0, 0],
                [-9.98334166e-02, 9.95004165e-01, 0, 0],
            ]
        ]
        assert math.allclose(rep2.A, A_exp)
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1])

    def test_trainable_parameters(self):
        gate1 = BSgate([0, 1], 1, 1)
        gate2 = BSgate([0, 1], 1, 1, theta_trainable=True, theta_bounds=(-2, 2))
        gate3 = BSgate([0, 1], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.theta.value = 3

        gate2.theta.value = 2
        assert gate2.theta.value == 2

        gate3.phi.value = 2
        assert gate3.phi.value == 2
