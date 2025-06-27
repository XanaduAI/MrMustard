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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.lab.states import Coherent, Vacuum
from mrmustard.lab.transformations import (
    Attenuator,
    Channel,
    Dgate,
    Identity,
    Map,
    Operation,
    PhaseNoise,
    Sgate,
    Unitary,
)
from mrmustard.physics.wires import Wires

from ...random import Abc_triple


class TestOperation:
    r"""
    Tests the Operation class.
    """

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init_from_bargmann(self, batch_shape):
        A, b, c = Abc_triple(3, batch_shape)
        operator = Operation.from_bargmann((0,), (1, 2), (A, b, c), "my_operator")
        assert math.allclose(operator.ansatz.A, A)
        assert math.allclose(operator.ansatz.b, b)
        assert math.allclose(operator.ansatz.c, c)


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
        unitary1 = Dgate(0, 1) >> Dgate(1, 1)
        unitary2 = Dgate(1, 2) >> Dgate(2, 2)
        u_component = CircuitComponent(unitary1.ansatz, unitary1.wires, unitary1.name)
        channel = Attenuator(1, 1)
        ch_component = CircuitComponent(channel.ansatz, channel.wires, channel.name)

        assert isinstance(unitary1 >> unitary2, Unitary)
        assert isinstance(unitary1 >> channel, Channel)
        assert isinstance(unitary1 >> u_component, CircuitComponent)
        assert isinstance(unitary1 >> ch_component, CircuitComponent)

    def test_repr(self):
        unitary1 = Dgate(0, 1)
        u_component = CircuitComponent(unitary1.ansatz, unitary1.wires, unitary1.name)
        assert repr(unitary1) == "Dgate(modes=(0,), name=Dgate, repr=PolyExpAnsatz)"
        assert repr(unitary1.to_fock(5)) == "Dgate(modes=(0,), name=Dgate, repr=ArrayAnsatz)"
        assert repr(u_component) == "CircuitComponent(modes=(0,), name=Dgate, repr=PolyExpAnsatz)"
        assert (
            repr(u_component.to_fock(5))
            == "CircuitComponent(modes=(0,), name=Dgate, repr=ArrayAnsatz)"
        )

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init_from_bargmann(self, batch_shape):
        A, b, c = Abc_triple(2, batch_shape)
        gate = Unitary.from_bargmann((2,), (2,), (A, b, c), "my_unitary")
        assert math.allclose(gate.ansatz.A, A)
        assert math.allclose(gate.ansatz.b, b)
        assert math.allclose(gate.ansatz.c, c)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init_from_symplectic(self, batch_shape):
        S = math.random_symplectic(2)
        S = math.broadcast_to(S, batch_shape + S.shape)
        u = Unitary.from_symplectic((0, 1), S)
        assert u.ansatz.batch_shape == batch_shape
        assert u.contract(u.dual, "zip") == Identity((0, 1))
        assert u.dual.contract(u, "zip") == Identity((0, 1))

    def test_init_from_fock(self):
        cutoff = 100
        eigs = [math.exp(1j * n**2) for n in range(cutoff)]
        kerr = Unitary.from_fock((0,), (0,), math.diag(math.astensor(eigs)))

        assert math.allclose((kerr >> kerr.dual).fock_array(), math.eye(cutoff))

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_inverse_unitary(self, batch_shape):
        r = math.broadcast_to(0.1, batch_shape)
        phi = math.broadcast_to(0.2, batch_shape)
        u = Sgate(0, r, phi).contract(Dgate(0, r, phi), "zip")
        gate = Unitary(u.ansatz, u.wires, u.name)
        gate_inv = gate.inverse()
        gate_inv_inv = gate_inv.inverse()
        assert gate_inv_inv == gate
        should_be_identity = gate >> gate_inv
        assert should_be_identity.ansatz == Dgate(0, 0.0, 0.0).ansatz

    def test_random(self):
        modes = (1, 3, 20)
        u = Unitary.random(modes)
        assert (u >> u.dual) == Identity(modes)


class TestMap:
    r"""
    Tests the Map class.
    """

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init_from_bargmann(self, batch_shape):
        A, b, c = Abc_triple(4, batch_shape)
        my_map = Map.from_bargmann((0,), (0,), (A, b, c), "my_map")
        assert math.allclose(my_map.ansatz.A, A)
        assert math.allclose(my_map.ansatz.b, b)
        assert math.allclose(my_map.ansatz.c, c)


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

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init_from_bargmann(self, batch_shape):
        A, b, c = Abc_triple(4, batch_shape)
        channel = Channel.from_bargmann((0,), (0,), (A, b, c), "my_channel")
        assert math.allclose(channel.ansatz.A, A)
        assert math.allclose(channel.ansatz.b, b)
        assert math.allclose(channel.ansatz.c, c)

    def test_rshift(self):
        unitary = Dgate(0, 1) >> Dgate(1, 1)
        u_component = CircuitComponent(unitary.ansatz, unitary.wires, unitary.name)
        channel1 = Attenuator(1, 0.9) >> Attenuator(2, 0.9)
        channel2 = Attenuator(2, 0.9) >> Attenuator(3, 0.9)
        ch_component = CircuitComponent(channel1.ansatz, channel1.wires, channel1.name)

        assert isinstance(channel1 >> unitary, Channel)
        assert isinstance(channel1 >> channel2, Channel)
        assert isinstance(channel1 >> u_component, CircuitComponent)
        assert isinstance(channel1 >> ch_component, CircuitComponent)

    def test_repr(self):
        channel1 = Attenuator(0, 0.9)
        ch_component = CircuitComponent(channel1.ansatz, channel1.wires, channel1.name)

        assert repr(channel1) == "Attenuator(modes=(0,), name=Att~, repr=PolyExpAnsatz)"
        assert repr(ch_component) == "CircuitComponent(modes=(0,), name=Att~, repr=PolyExpAnsatz)"

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_inverse_channel(self, batch_shape):
        r = math.broadcast_to(0.1, batch_shape)
        phi = math.broadcast_to(0.2, batch_shape)
        g = Sgate(0, r, phi).contract(Dgate(0, r, phi), "zip").contract(Attenuator(0, 0.5), "zip")
        gate = Channel(g.ansatz, g.wires, g.name)
        should_be_identity = gate >> gate.inverse()
        assert should_be_identity.ansatz == Attenuator(0, 1.0).ansatz

    def test_random(self):
        modes = (1, 2, 6)
        assert math.allclose((Vacuum(modes) >> Channel.random(modes)).probability, 1)

    @pytest.mark.parametrize("modes", [(0,), (0, 1), (0, 1, 2)])
    def test_is_CP(self, modes):
        u = Unitary.random(modes).ansatz
        assert Channel.from_ansatz(modes, modes, u.conj & u).is_CP

    def test_is_TP(self):
        assert Attenuator(0, 0.5).is_CP

    def test_is_physical(self):
        assert Channel.random(range(5)).is_physical

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_XY(self, batch_shape):
        U = Unitary.random((0, 1))
        u = U.ansatz
        unitary_channel = Channel.from_ansatz((0, 1), (0, 1), u.conj & u)
        X, Y = unitary_channel.XY
        assert math.allclose(X, U.symplectic) and math.allclose(Y, math.zeros((4, 4)))

        transmissivity = math.broadcast_to(0.2, batch_shape)
        X, Y = Attenuator(0, transmissivity).XY
        expected_X = math.broadcast_to(np.sqrt(0.2), (*batch_shape, 2, 2)) * np.eye(2)
        expected_Y = math.broadcast_to(0.4, (*batch_shape, 2, 2)) * np.eye(2)
        assert math.allclose(X, expected_X) and math.allclose(Y, expected_Y)

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_from_XY(self, nmodes):
        X = settings.rng.random((2 * nmodes, 2 * nmodes))
        Y = settings.rng.random((2 * nmodes, 2 * nmodes))
        x, y = Channel.from_XY(tuple(range(nmodes)), tuple(range(nmodes)), X, Y).XY
        assert math.allclose(x, X)
        assert math.allclose(y, Y)

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_from_XY_batched(self, nmodes):
        ch1 = Channel.random(list(range(nmodes))) + Channel.random(list(range(nmodes)))
        X, Y = ch1.XY

        ch2 = Channel.from_XY(tuple(range(nmodes)), tuple(range(nmodes)), X, Y)
        assert ch1 == ch2

    def test_from_fock(self):
        # Here we test our from_fock method by a PhaseNoise example
        cutoff = 6
        ph_n = np.zeros((cutoff, cutoff, cutoff, cutoff))
        sigma = 1

        for m in range(cutoff):
            for n in range(cutoff):
                ph_n[m, m, n, n] = math.exp(-0.5 * (m - n) ** 2 * sigma**2)

        phi = Channel.from_fock((0,), (0,), ph_n)
        psi = Coherent(0, 2) >> phi

        assert psi.to_fock((cutoff, cutoff)) == (Coherent(0, 2) >> PhaseNoise(0, sigma)).to_fock(
            (cutoff, cutoff),
        )
