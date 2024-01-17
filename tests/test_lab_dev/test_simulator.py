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
import pytest

from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.simulator import Simulator
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Dgate


class TestSimulator:
    r"""
    Tests for the simulator class.
    """

    @pytest.mark.parametrize("modes", [[0], [1], [2]])
    @pytest.mark.parametrize("add_bras", [True, False])
    def test_simulate_one_mode_components_in_series(self, modes, add_bras):
        r"""
        Simulates a circuit with one-mode Dgates applied in series.
        """
        d1 = Dgate(1, modes=modes)
        d2 = Dgate(2, modes=modes)

        circuit = Circuit([d1, d1, d2])
        result = Simulator().run(circuit, add_bras)

        rep = result.representation
        A = [[0, 1], [1, 0]]
        b = [4, -4]

        assert result.modes == modes
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()

    @pytest.mark.parametrize("add_bras", [True, False])
    def test_simulate_one_mode_components_in_parallel(self, add_bras):
        r"""
        Simulates a circuit with one-mode Dgates applied in parallel.
        """
        d1 = Dgate(1, modes=[1])
        d2 = Dgate(2, modes=[2])
        d3 = Dgate(3, modes=[3])
        d4 = Dgate(4, modes=[4])

        circuit = Circuit([d3, d1, d4, d2])
        result = Simulator().run(circuit, add_bras)

        rep = result.representation
        A = np.kron(np.eye(4), [[0, 1], [1, 0]])
        b = [1, -1, 2, -2, 3, -3, 4, -4]

        assert result.modes == [1, 2, 3, 4]
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()

    @pytest.mark.parametrize("add_bras", [True, False])
    def test_simulate_one_mode_components_in_parallel_and_series(self, add_bras):
        r"""
        Simulates a circuit with one-mode Dgates applied in parallel and in series.
        """
        d1 = Dgate(1, modes=[1])
        d2 = Dgate(2, modes=[2])
        d3 = Dgate(3, modes=[3])
        d4 = Dgate(4, modes=[4])

        circuit = Circuit([d1, d1, d2, d1, d3, d1, d4, d2, d1])
        result = Simulator().run(circuit, add_bras)

        rep = result.representation
        A = np.kron(np.eye(4), [[0, 1], [1, 0]])
        b = [5, -5, 4, -4, 3, -3, 4, -4]

        assert result.modes == [1, 2, 3, 4]
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()


class TestSimulatorBroken:
    r"""
    Tests for the simulator class that currently do not pass.
    """

    @pytest.mark.parametrize("modes", [[0, 1]])
    def test_simulate_multi_mode_components_in_series(self, modes):
        r"""
        Simulates a circuit with multi-mode Dgates applied in series.
        """
        d1 = Dgate(1, modes=modes)
        d2 = Dgate(2, modes=modes)
        circuit = Circuit([d1, d1, d2])

        result = Simulator().run(circuit)
        rep = result.representation
        A = [[0, 1], [1, 0]]
        b = [4, -4]

        assert result.modes == modes
        assert result.name == ""
        assert (rep.A == A).all()
        assert (rep.b == b).all()
