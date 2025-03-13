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

"""Tests for the ``Dgate`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev import Dgate, SqueezedVacuum
from mrmustard.physics.ansatz import ArrayAnsatz


class TestDgate:
    r"""
    Tests for the ``Dgate`` class.
    """

    modes = [0, 1, 7]
    x = [1, 2, 3]
    y = [4, 5, 6]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        gate = Dgate(modes, x, y)

        assert gate.name == "Dgate"
        assert gate.modes == (modes,)

    def test_to_fock_method(self):
        # test stable Dgate in fock basis
        state = SqueezedVacuum(0, r=1.0)
        # displacement gate in fock representation for large displacement
        dgate = Dgate(0, x=10.0).to_fock(150)
        assert (state.to_fock() >> dgate).probability < 1
        assert np.all(math.abs(dgate.fock_array(150)) < 1)

    def test_representation(self):
        rep1 = Dgate(mode=0, x=0.1, y=0.1).ansatz
        assert math.allclose(rep1.A, [[[0, 1], [1, 0]]])
        assert math.allclose(rep1.b, [[0.1 + 0.1j, -0.1 + 0.1j]])
        assert math.allclose(rep1.c, [0.990049833749168])

        rep2 = Dgate(mode=2, x=0.1, y=0.2).ansatz
        assert math.allclose(rep1.A, [[[0, 1], [1, 0]]])
        assert math.allclose(rep2.b, [[0.1 + 0.2j, -0.1 + 0.2j]])
        assert math.allclose(rep2.c, [0.97530991 + 0.0j])

    def test_trainable_parameters(self):
        gate1 = Dgate(0, 1, 1)
        gate2 = Dgate(0, 1, 1, x_trainable=True, x_bounds=(-2, 2))
        gate3 = Dgate(0, 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.x.value = 3

        gate2.parameters.x.value = 2
        assert gate2.parameters.x.value == 2

        gate3.parameters.y.value = 2
        assert gate3.parameters.y.value == 2

        gate_fock = gate3.to_fock()
        assert isinstance(gate_fock.ansatz, ArrayAnsatz)
        assert gate_fock.parameters.y.value == 2
