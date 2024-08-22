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

"""Tests for the base transformation subpackage."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.transformations import (
    Attenuator,
    Channel,
    Dgate,
    Sgate,
    Identity,
    Unitary,
    Operation,
)
from mrmustard.lab_dev.wires import Wires
from mrmustard.lab_dev.states import Vacuum


class TestOperation:
    r"""
    Tests the Operation class.
    """

    def test_init_from_bargmann(self):
        A = np.array([[0, 1, 2], [1, 0, 0], [0, 4, 2]])
        b = np.array([0, 1, 5])
        c = 1
        operator = Operation.from_bargmann([0], [1, 2], (A, b, c), "my_operator")
        assert np.allclose(operator.representation.A[None, ...], A)
        assert np.allclose(operator.representation.b[None, ...], b)


class TestUnitary:
    r"""
    Tests for the ``Unitary`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_unitary"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        gate = Unitary(modes, modes, name=name)

        assert gate.name[:1] == (name or "U")[:1]
        assert list(gate.modes) == sorted(modes)
        assert gate.wires == Wires(modes_in_ket=modes, modes_out_ket=modes)

    def test_rshift(self):
        unitary1 = Dgate([0, 1], 1)
        unitary2 = Dgate([1, 2], 2)
        u_component = CircuitComponent._from_attributes(
            unitary1.representation, unitary1.wires, unitary1.name
        )  # pylint: disable=protected-access
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent._from_attributes(
            channel.representation, channel.wires, channel.name
        )  # pylint: disable=protected-access

        assert isinstance(unitary1 >> unitary2, Unitary)
        assert isinstance(unitary1 >> channel, Channel)
        assert isinstance(unitary1 >> u_component, CircuitComponent)
        assert isinstance(unitary1 >> ch_component, CircuitComponent)

    def test_repr(self):
        unitary1 = Dgate([0, 1], 1)
        u_component = CircuitComponent._from_attributes(
            unitary1.representation, unitary1.wires, unitary1.name
        )  # pylint: disable=protected-access
        assert repr(unitary1) == "Dgate(modes=[0, 1], name=Dgate, repr=Bargmann)"
        assert repr(unitary1.to_fock(5)) == "Dgate(modes=[0, 1], name=Dgate, repr=Fock)"
        assert repr(u_component) == "CircuitComponent(modes=[0, 1], name=Dgate, repr=Bargmann)"
        assert (
            repr(u_component.to_fock(5)) == "CircuitComponent(modes=[0, 1], name=Dgate, repr=Fock)"
        )

    def test_init_from_bargmann(self):
        A = np.array([[0, 1], [1, 0]])
        b = np.array([0, 0])
        c = 1
        gate = Unitary.from_bargmann([2], [2], (A, b, c), "my_unitary")
        assert np.allclose(gate.representation.A[None, ...], A)
        assert np.allclose(gate.representation.b[None, ...], b)

    def test_init_from_symplectic(self):
        S = math.random_symplectic(2)
        u = Unitary.from_symplectic([0, 1], S)
        assert u >> u.dual == Identity([0, 1])
        assert u.dual >> u == Identity([0, 1])

    def test_inverse_unitary(self):
        gate = Sgate([0], 0.1, 0.2) >> Dgate([0], 0.1, 0.2)
        gate_inv = gate.inverse()
        gate_inv_inv = gate_inv.inverse()
        assert gate_inv_inv == gate
        should_be_identity = gate >> gate_inv
        assert should_be_identity.representation == Dgate([0], 0.0, 0.0).representation

    def test_random(self):
        modes = [3, 1, 20]
        u = Unitary.random(modes)
        assert (u >> u.dual) == Identity(modes)


class TestChannel:
    r"""
    Tests for the ``Channel`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_channel"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        gate = Channel(modes, modes, name=name)

        assert gate.name[:2] == (name or "Ch")[:2]
        assert list(gate.modes) == sorted(modes)
        assert gate.wires == Wires(
            modes_out_bra=modes,
            modes_in_bra=modes,
            modes_out_ket=modes,
            modes_in_ket=modes,
        )

    def test_init_from_bargmann(self):
        A = np.arange(16).reshape(4, 4)
        b = np.array([0, 1, 2, 3])
        c = 1
        channel = Channel.from_bargmann([0], [0], (A, b, c), "my_channel")
        assert np.allclose(channel.representation.A[None, ...], A)
        assert np.allclose(channel.representation.b[None, ...], b)

    def test_rshift(self):
        unitary = Dgate([0, 1], 1)
        u_component = CircuitComponent._from_attributes(
            unitary.representation, unitary.wires, unitary.name
        )  # pylint: disable=protected-access
        channel1 = Attenuator([1, 2], 0.9)
        channel2 = Attenuator([2, 3], 0.9)
        ch_component = CircuitComponent._from_attributes(
            channel1.representation, channel1.wires, channel1.name
        )  # pylint: disable=protected-access

        assert isinstance(channel1 >> unitary, Channel)
        assert isinstance(channel1 >> channel2, Channel)
        assert isinstance(channel1 >> u_component, CircuitComponent)
        assert isinstance(channel1 >> ch_component, CircuitComponent)

    def test_repr(self):
        channel1 = Attenuator([0, 1], 0.9)
        ch_component = CircuitComponent._from_attributes(
            channel1.representation, channel1.wires, channel1.name
        )  # pylint: disable=protected-access

        assert repr(channel1) == "Attenuator(modes=[0, 1], name=Att, repr=Bargmann)"
        assert repr(ch_component) == "CircuitComponent(modes=[0, 1], name=Att, repr=Bargmann)"

    def test_inverse_channel(self):
        gate = Sgate([0], 0.1, 0.2) >> Dgate([0], 0.1, 0.2) >> Attenuator([0], 0.5)
        should_be_identity = gate >> gate.inverse()
        assert should_be_identity.representation == Attenuator([0], 1.0).representation

    def test_random(self):

        modes = [2, 6, 1]
        assert np.isclose((Vacuum(modes) >> Channel.random(modes)).probability, 1)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [0, 1, 2]])
    def test_is_CP(self, modes):
        u = Unitary.random(modes).representation
        kraus = u @ u.conj()
        assert Channel.from_bargmann(modes, modes, kraus.triple).is_CP

    def test_is_TP(self):
        assert Attenuator([0, 1], 0.5).is_CP

    def test_is_physical(self):
        assert Channel.random(range(5)).is_physical

    def test_XY(self):
        U = Unitary.random([0, 1])
        u = U.representation
        unitary_channel = Channel.from_bargmann([0, 1], [0, 1], (u.conj() @ u).triple)
        X, Y = unitary_channel.XY
        assert np.allclose(X, U.symplectic) and np.allclose(Y, np.zeros(4))

        X, Y = Attenuator([0], 0.2).XY
        assert np.allclose(X, np.sqrt(0.2) * np.eye(2)) and np.allclose(Y, 0.4 * np.eye(2))
