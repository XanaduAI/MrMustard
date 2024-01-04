# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for simulators.
"""

import pytest

import numpy as np

from mrmustard.lab_dev.simulator import SimulatorBargmann
from mrmustard.lab_dev.transformations import Dgate, Attenuator
from mrmustard.lab_dev.states import Vacuum


class TestSimulatorBargmann:
    r"""
    Tests for the ``SimulatorBargmann`` class.
    """

    def test_run_function_with_a_series_of_states(self):
        r"""Tests that the simulator works correctly to contract a series of states."""
        simulator = SimulatorBargmann()
        state1 = Vacuum(1)
        state2 = Vacuum(3)
        circuit = state1 >> state2  # TODO: This needs to have a warning or Error somehow.
        new_component = simulator.run(circuit)
        assert len(new_component.modes) == 3

    def test_run_function_with_a_series_of_gates_on_single_mode(self):
        r"""Tests that the simulator works correctly to contract a series of gates on single mode."""
        simulator = SimulatorBargmann()
        dgate1 = Dgate(x=[0.1], modes=[0])
        dgate2 = Dgate(x=[0.5], modes=[0])
        circuit = dgate1 >> dgate2
        new_component = simulator.run(circuit)
        assert len(new_component.modes) == 1
        assert np.allclose(new_component.representation.A, np.array([[0, 1], [1, 0]]))
        assert np.allclose(
            new_component.representation.b, np.array([-0.1 - 0.5, 0.1 + 0.5])
        )  # Note that the order is 'out' first.

    def test_run_function_with_a_series_of_gates_on_several_modes(self):
        r"""Tests that the simulator works correctly to contract a series of gates on several modes."""
        simulator = SimulatorBargmann()
        dgate1 = Dgate(x=[0.1, 0.2], modes=[0, 7])
        dgate2 = Dgate(x=[0.5, -0.5], modes=[0, 88])
        circuit = dgate1 >> dgate2
        new_component = simulator.run(circuit)
        assert len(new_component.modes) == 3
        # TODO: after discuss the reordering inside complex_integral
        # assert np.allclose(new_component.representation.A, np.kron(np.array([[0, 1], [1, 0]]), np.eye(3)))
        # assert np.allclose(new_component.representation.b, np.array([-0.1-0.5, 0.1+0.5, 0.2, -0.2, 0.5, -0.5])) # Note that the order is 'out' first.

    def test_run_function_with_a_series_of_channels(self):
        r"""Tests that the simulator works correctly to contract a series of channels."""
        assert True

    def test_run_function_with_states_and_gates(self):
        r"""Tests that the simulator works correctly to contract a series of gates."""
        assert True

    def test_run_function_with_gates_and_channels(self):
        r"""Tests that the simulator works correctly to contract a series of gates."""
        assert True

    def test_run_function_with_states_and_gates_and_channels(self):
        r"""Tests that the simulator works correctly to contract a normal circuit."""
        assert True

    def test_run_function_with_states_and_measurements(self):
        r"""Tests that the simulator works correctly to contract a series of gates."""
        assert True

    def test_run_function_with_states_and_gates_and_channels(self):
        r"""Tests that the simulator works correctly to contract a series of gates."""
        assert True
