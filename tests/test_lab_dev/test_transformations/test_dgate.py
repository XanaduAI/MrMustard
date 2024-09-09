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

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import pytest

from mrmustard import math
from mrmustard.lab_dev import Dgate, SqueezedVacuum
from mrmustard.physics.representations import Fock


class TestDgate:
    r"""
    Tests for the ``Dgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        gate = Dgate(modes, x, y)

        assert gate.name == "Dgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="x"):
            Dgate(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="y"):
            Dgate(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_to_fock_method(self):
        # test stable Dgate in fock basis
        state = SqueezedVacuum([0], r=1.0)
        # displacement gate in fock representation for large displacement
        dgate = Dgate([0], x=10.0).to_fock(150)
        assert ((state.to_fock() >> dgate).probability < 1) and all(math.abs(dgate.fock(150) < 1))

    def test_representation(self):
        rep1 = Dgate(modes=[0], x=0.1, y=0.1).representation
        assert math.allclose(rep1.A, [[[0, 1], [1, 0]]])
        assert math.allclose(rep1.b, [[0.1 + 0.1j, -0.1 + 0.1j]])
        assert math.allclose(rep1.c, [0.990049833749168])

        rep2 = Dgate(modes=[0, 1], x=[0.1, 0.2], y=0.1).representation
        assert math.allclose(rep2.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]])
        assert math.allclose(rep2.b, [[0.1 + 0.1j, 0.2 + 0.1j, -0.1 + 0.1j, -0.2 + 0.1j]])
        assert math.allclose(rep2.c, [0.9656054162575665])

        rep3 = Dgate(modes=[1, 8], x=[0.1, 0.2]).representation
        assert math.allclose(rep3.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]])
        assert math.allclose(rep3.b, [[0.1, 0.2, -0.1, -0.2]])
        assert math.allclose(rep3.c, [0.9753099120283327])

    def test_trainable_parameters(self):
        gate1 = Dgate([0], 1, 1)
        gate2 = Dgate([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        gate3 = Dgate([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.x.value = 3

        gate2.x.value = 2
        assert gate2.x.value == 2

        gate3.y.value = 2
        assert gate3.y.value == 2

        gate_fock = gate3.to_fock()
        assert isinstance(gate_fock.representation, Fock)
        assert gate_fock.y.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Dgate(modes=[0], x=[0.1, 0.2]).representation
