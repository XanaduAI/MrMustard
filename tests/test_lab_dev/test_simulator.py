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

"""Tests for the Simulator class."""

import numpy as np

from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.simulator import Simulator
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Dgate


class TestSimulator:
    r"""
    Tests for the simulator class.
    """

    def test_state_plus_gates(self):
        r"""
        Simulates a circuit with a three-mode vacuum state undergoing one- and two-mode Dgates.
        """
        vac = Vacuum(3)
        d0 = Dgate(1, modes=[0])
        d01 = Dgate([2, 3], modes=[0, 1])
        d1 = Dgate(4, modes=[1])
        d2 = Dgate(5, modes=[2])

        circuit = Circuit([vac, d1, d01, d0, d2, d0])
        result = Simulator().run(circuit, add_bras=False)

        rep = result.representation
        A = [[0, 0, 0]] * 3
        b = [4, 7, 5]

        assert result.modes == [0, 1, 2]
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()

    def test_gates_only(self):
        r"""
        Simulates a circuit with a sequence of one- and two-mode Dgates.
        """
        d0 = Dgate(1, modes=[0])
        d1 = Dgate(2, modes=[1])
        d01 = Dgate(3, modes=[0, 1])
        d02 = Dgate(4, modes=[0, 2])

        circuit = Circuit([d1, d0, d02, d01, d01])
        result = Simulator().run(circuit, add_bras=False)

        rep = result.representation
        A = np.kron([[0, 1], [1, 0]], np.eye(3))
        b = [11, 8, 4, -11, -8, -4]

        assert result.modes == [0, 1, 2]
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()

    def test_add_bras(self):
        r"""
        Simulates a circuit with one-mode Dgates applied in parallel and in series, with ``add_bras=True``.
        """
        d1 = Dgate(1, modes=[1])
        d2 = Dgate(2, modes=[2])
        d3 = Dgate(3, modes=[3])
        d4 = Dgate(4, modes=[4])

        circuit = Circuit([d1, d1, d2, d1, d3, d1, d4, d2, d1])
        result = Simulator().run(circuit, add_bras=True)

        rep = result.representation
        A = np.kron(np.eye(2), np.kron([[0, 1], [1, 0]], np.eye(4)))
        b = [5, 4, 3, 4, -5, -4, -3, -4] * 2

        assert result.modes == [1, 2, 3, 4]
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()
