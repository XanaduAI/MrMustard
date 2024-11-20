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

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import (
    Attenuator,
    Channel,
    Dgate,
    Identity,
    Map,
    Operation,
    Sgate,
    Unitary,
)
from mrmustard.physics.wires import Wires


class TestOperation:
    r"""
    Tests the Operation class.
    """

    def test_init_from_bargmann(self):
        A = np.array([[0, 1, 2], [1, 0, 0], [0, 4, 2]])
        b = np.array([0, 1, 5])
        c = 1
        operator = Operation.from_bargmann([0], [1, 2], (A, b, c), "my_operator")
        assert np.allclose(operator.ansatz.A[None, ...], A)
        assert np.allclose(operator.ansatz.b[None, ...], b)


class TestUnitary:
    r"""
    Tests for the ``Unitary`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_unitary"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        gate = Unitary.from_ansatz(modes, modes, name=name)

        assert gate.name[:1] == (name or "U")[:1]
        assert list(gate.modes) == sorted(modes)
        assert gate.wires == Wires(modes_in_ket=modes, modes_out_ket=modes)

    def test_rshift(self):
        unitary1 = Dgate([0, 1], 1)
        unitary2 = Dgate([1, 2], 2)
        u_component = CircuitComponent(unitary1.representation, unitary1.name)
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent(channel.representation, channel.name)

        assert isinstance(unitary1 >> unitary2, Unitary)
        assert isinstance(unitary1 >> channel, Channel)
        assert isinstance(unitary1 >> u_component, CircuitComponent)
        assert isinstance(unitary1 >> ch_component, CircuitComponent)

    def test_repr(self):
        unitary1 = Dgate([0, 1], 1)
        u_component = CircuitComponent(unitary1.representation, unitary1.name)
        assert repr(unitary1) == "Dgate(modes=[0, 1], name=Dgate, repr=PolyExpAnsatz)"
        assert repr(unitary1.to_fock(5)) == "Dgate(modes=[0, 1], name=Dgate, repr=ArrayAnsatz)"
        assert repr(u_component) == "CircuitComponent(modes=[0, 1], name=Dgate, repr=PolyExpAnsatz)"
        assert (
            repr(u_component.to_fock(5))
            == "CircuitComponent(modes=[0, 1], name=Dgate, repr=ArrayAnsatz)"
        )

    def test_init_from_bargmann(self):
        A = np.array([[0, 1], [1, 0]])
        b = np.array([0, 0])
        c = 1
        gate = Unitary.from_bargmann([2], [2], (A, b, c), "my_unitary")
        assert np.allclose(gate.ansatz.A[None, ...], A)
        assert np.allclose(gate.ansatz.b[None, ...], b)

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
        assert should_be_identity.ansatz == Dgate([0], 0.0, 0.0).ansatz

    def test_random(self):
        modes = [1, 3, 20]
        u = Unitary.random(modes)
        assert (u >> u.dual) == Identity(modes)


class TestMap:
    r"""
    Tests the Map class.
    """

    def test_init_from_bargmann(self):
        A = np.arange(16).reshape(4, 4)
        b = np.array([0, 1, 2, 3])
        c = 1
        map = Map.from_bargmann([0], [0], (A, b, c), "my_map")
        assert np.allclose(map.ansatz.A[None, ...], A)
        assert np.allclose(map.ansatz.b[None, ...], b)


class TestChannel:
    r"""
    Tests for the ``Channel`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_channel"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        gate = Channel.from_ansatz(modes, modes, name=name)

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
        assert np.allclose(channel.ansatz.A[None, ...], A)
        assert np.allclose(channel.ansatz.b[None, ...], b)

    def test_rshift(self):
        unitary = Dgate([0, 1], 1)
        u_component = CircuitComponent(unitary.representation, unitary.name)
        channel1 = Attenuator([1, 2], 0.9)
        channel2 = Attenuator([2, 3], 0.9)
        ch_component = CircuitComponent(channel1.representation, channel1.name)

        assert isinstance(channel1 >> unitary, Channel)
        assert isinstance(channel1 >> channel2, Channel)
        assert isinstance(channel1 >> u_component, CircuitComponent)
        assert isinstance(channel1 >> ch_component, CircuitComponent)

    def test_repr(self):
        channel1 = Attenuator([0, 1], 0.9)
        ch_component = CircuitComponent(channel1.representation, channel1.name)

        assert repr(channel1) == "Attenuator(modes=[0, 1], name=Att~, repr=PolyExpAnsatz)"
        assert repr(ch_component) == "CircuitComponent(modes=[0, 1], name=Att~, repr=PolyExpAnsatz)"

    def test_inverse_channel(self):
        gate = Sgate([0], 0.1, 0.2) >> Dgate([0], 0.1, 0.2) >> Attenuator([0], 0.5)
        should_be_identity = gate >> gate.inverse()
        assert should_be_identity.ansatz == Attenuator([0], 1.0).ansatz

    def test_random(self):
        modes = [1, 2, 6]
        assert np.isclose((Vacuum(modes) >> Channel.random(modes)).probability, 1)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [0, 1, 2]])
    def test_is_CP(self, modes):
        u = Unitary.random(modes).ansatz
        kraus = u @ u.conj
        assert Channel.from_bargmann(modes, modes, kraus.triple).is_CP

    def test_is_TP(self):
        assert Attenuator([0, 1], 0.5).is_CP

    def test_is_physical(self):
        assert Channel.random(range(5)).is_physical

    def test_XY(self):
        U = Unitary.random([0, 1])
        u = U.ansatz
        unitary_channel = Channel.from_bargmann([0, 1], [0, 1], (u.conj @ u).triple)
        X, Y = unitary_channel.XY
        assert np.allclose(X, U.symplectic) and np.allclose(Y, np.zeros(4))

        X, Y = Attenuator([0], 0.2).XY
        assert np.allclose(X, np.sqrt(0.2) * np.eye(2)) and np.allclose(Y, 0.4 * np.eye(2))

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_from_XY(self, nmodes):
        X = np.random.random((2 * nmodes, 2 * nmodes))
        Y = np.random.random((2 * nmodes, 2 * nmodes))
        x, y = Channel.from_XY(range(nmodes), range(nmodes), X, Y).XY
        assert math.allclose(x, X)
        assert math.allclose(y, Y)
