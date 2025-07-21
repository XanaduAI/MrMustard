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

"""Tests for the ``Rgate`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.transformations import Rgate


class TestRgate:
    r"""
    Tests for the ``Rgate`` class.
    """

    modes = [0, 1, 7]
    thetas = [1, 2, 3]

    @pytest.mark.parametrize("modes,theta", zip(modes, thetas))
    def test_init(self, modes, theta):
        gate = Rgate(modes, theta)

        assert gate.name == "Rgate"
        assert gate.modes == (modes,)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        theta = math.broadcast_to(0.1, batch_shape)
        rep1 = Rgate(mode=0, theta=theta).ansatz
        assert math.allclose(
            rep1.A,
            [
                [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                [0.99500417 + 0.09983342j, 0.0 + 0.0j],
            ],
        )
        assert math.allclose(rep1.b, math.zeros((2,)))
        assert math.allclose(rep1.c, 1.0 + 0.0j)

        rep2 = (Rgate(mode=0, theta=theta) >> Rgate(mode=1, theta=0.3)).ansatz
        assert math.allclose(
            rep2.A,
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.95533649 + 0.29552021j],
                [0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.95533649 + 0.29552021j, 0.0 + 0.0j, 0.0 + 0.0j],
            ],
        )
        assert math.allclose(rep2.b, math.zeros((4,)))
        assert math.allclose(rep2.c, 1.0 + 0.0j)

        rep3 = Rgate(mode=1, theta=theta).ansatz
        assert math.allclose(
            rep3.A,
            [
                [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                [0.99500417 + 0.09983342j, 0.0 + 0.0j],
            ],
        )
        assert math.allclose(rep3.b, math.zeros((2,)))
        assert math.allclose(rep3.c, 1.0 + 0.0j)

    def test_trainable_parameters(self):
        gate1 = Rgate(0, 1)
        gate2 = Rgate(0, 1, True, (-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.theta.value = 3

        gate2.parameters.theta.value = 2
        assert gate2.parameters.theta.value == 2
