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

import pytest

from mrmustard import math
from mrmustard.lab.transformations import Sgate


class TestSgate:
    r"""
    Tests for the ``Sgate`` class.
    """

    modes = [0, 1, 7]
    r = [1, 2, 3]
    phi = [3, 4, 5]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        gate = Sgate(modes, r, phi)

        assert gate.name == "Sgate"
        assert gate.modes == (modes,)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        r = math.broadcast_to(0.1, batch_shape)
        phi = math.broadcast_to(0.2, batch_shape)
        rep1 = Sgate(mode=0, r=r, phi=phi).ansatz
        assert math.allclose(
            rep1.A,
            [
                [-0.09768127 - 1.98009738e-02j, 0.99502075],
                [0.99502075, 0.09768127 - 0.01980097j],
            ],
        )
        assert math.allclose(rep1.b, math.zeros((2,)))
        assert math.allclose(rep1.c, 0.9975072676192522)

        rep2 = (Sgate(mode=0, r=r, phi=phi) >> Sgate(mode=1, r=0.3, phi=0.2)).ansatz
        assert math.allclose(
            rep2.A,
            [
                [-0.09768127 - 1.98009738e-02j, 0, 0.99502075, 0],
                [0, -0.28550576 - 5.78748818e-02j, 0, 0.95662791],
                [0.99502075, 0, 0.09768127 - 1.98009738e-02j, 0],
                [0, 0.95662791, 0, 0.28550576 - 5.78748818e-02j],
            ],
        )
        assert math.allclose(rep2.b, math.zeros((4,)))
        assert math.allclose(rep2.c, 0.9756354961606032)

        rep3 = Sgate(mode=1, r=r).ansatz
        assert math.allclose(
            rep3.A,
            [
                [-0.09966799 + 0.0j, 0.99502075 + 0.0j],
                [0.99502075 + 0.0j, 0.09966799 + 0.0j],
            ],
        )
        assert math.allclose(rep3.b, math.zeros((2,)))
        assert math.allclose(rep3.c, 0.9975072676192522)

    def test_trainable_parameters(self):
        gate1 = Sgate(0, 1, 1)
        gate2 = Sgate(0, 1, 1, r_trainable=True, r_bounds=(-2, 2))
        gate3 = Sgate(0, 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.r.value = 3

        gate2.parameters.r.value = 2
        assert gate2.parameters.r.value == 2

        gate3.parameters.phi.value = 2
        assert gate3.parameters.phi.value == 2
