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

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import Rgate


class TestRgate:
    r"""
    Tests for the ``Rgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    phis = [[1], 1, [1, 2]]

    @pytest.mark.parametrize("modes,phi", zip(modes, phis))
    def test_init(self, modes, phi):
        gate = Rgate(modes, phi)

        assert gate.name == "Rgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="phi"):
            Rgate(modes=[0, 1], phi=[2, 3, 4])

    def test_representation(self):
        rep1 = Rgate(modes=[0], phi=0.1).ansatz
        assert math.allclose(
            rep1.A,
            [
                [
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [1.0 + 0.0j])

        rep2 = Rgate(modes=[0, 1], phi=[0.1, 0.3]).ansatz
        assert math.allclose(
            rep2.A,
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.95533649 + 0.29552021j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.95533649 + 0.29552021j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1.0 + 0.0j])

        rep3 = Rgate(modes=[1], phi=0.1).ansatz
        assert math.allclose(
            rep3.A,
            [
                [
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep3.b, np.zeros((1, 2)))
        assert math.allclose(rep3.c, [1.0 + 0.0j])

    def test_trainable_parameters(self):
        gate1 = Rgate([0], 1)
        gate2 = Rgate([0], 1, True, (-2, 2))

        with pytest.raises(AttributeError):
            gate1.phi.value = 3

        gate2.phi.value = 2
        assert gate2.phi.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Rgate(modes=[0], phi=[0.1, 0.2]).ansatz
