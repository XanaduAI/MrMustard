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

"""Tests for the ``Sgate`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import Sgate


class TestSgate:
    r"""
    Tests for the ``Sgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        gate = Sgate(modes, r, phi)

        assert gate.name == "Sgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="r"):
            Sgate(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="phi"):
            Sgate(modes=[0, 1], r=1, phi=[2, 3, 4])

    def test_representation(self):
        rep1 = Sgate(modes=[0], r=0.1, phi=0.2).ansatz
        assert math.allclose(
            rep1.A,
            [
                [
                    [-0.09768127 - 1.98009738e-02j, 0.99502075],
                    [0.99502075, 0.09768127 - 0.01980097j],
                ]
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [0.9975072676192522])

        rep2 = Sgate(modes=[0, 1], r=[0.1, 0.3], phi=0.2).ansatz
        assert math.allclose(
            rep2.A,
            [
                [
                    [-0.09768127 - 1.98009738e-02j, 0, 0.99502075, 0],
                    [0, -0.28550576 - 5.78748818e-02j, 0, 0.95662791],
                    [0.99502075, 0, 0.09768127 - 1.98009738e-02j, 0],
                    [0, 0.95662791, 0, 0.28550576 - 5.78748818e-02j],
                ]
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [0.9756354961606032])

        rep3 = Sgate(modes=[1], r=0.1).ansatz
        assert math.allclose(
            rep3.A,
            [
                [
                    [-0.09966799 + 0.0j, 0.99502075 + 0.0j],
                    [0.99502075 + 0.0j, 0.09966799 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep3.b, np.zeros((1, 2)))
        assert math.allclose(rep3.c, [0.9975072676192522])

    def test_trainable_parameters(self):
        gate1 = Sgate([0], 1, 1)
        gate2 = Sgate([0], 1, 1, r_trainable=True, r_bounds=(-2, 2))
        gate3 = Sgate([0], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.r.value = 3

        gate2.parameters.r.value = 2
        assert gate2.parameters.r.value == 2

        gate3.parameters.phi.value = 2
        assert gate3.parameters.phi.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Sgate(modes=[0], r=[0.1, 0.2]).ansatz
