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

"""Tests for the state subpackage."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.states import Coherent, DM, Ket, Number, Vacuum
from mrmustard.lab_dev.transformations import Attenuator, Dgate
from mrmustard.lab_dev.wires import Wires


class TestKet:
    r"""
    Tests for the ``Ket`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_ket"])
    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_init(self, name, modes):
        state = Ket(name, modes)

        assert state.name == (name if name else "")
        assert state.modes == sorted(modes)
        assert state.wires == Wires(modes_out_ket=modes)

    def test_rshift(self):
        ket = Coherent([0, 1], 1)
        unitary = Dgate([0], 1)
        u_component = CircuitComponent.from_attributes(
            unitary.name, unitary.representation, unitary.wires
        )
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent.from_attributes(
            channel.name, channel.representation, channel.wires
        )

        assert isinstance(ket >> unitary, Ket)
        assert isinstance(ket >> channel, DM)
        assert isinstance(ket >> unitary >> channel, DM)
        assert isinstance(ket >> channel >> unitary, DM)
        assert isinstance(ket >> u_component, CircuitComponent)
        assert isinstance(ket >> ch_component, CircuitComponent)

    def test_repr(self):
        ket = Coherent([0, 1], 1)
        ket_component = CircuitComponent.from_attributes(ket.name, ket.representation, ket.wires)

        assert repr(ket) == "Ket(name=Coherent, modes=[0, 1])"
        assert repr(ket_component) == "CircuitComponent(name=Coherent, modes=[0, 1])"


class TestDM:
    r"""
    Tests for the ``DM`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_dm"])
    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_init(self, name, modes):
        state = DM(name, modes)

        assert state.name == (name if name else "")
        assert state.modes == sorted(modes)
        assert state.wires == Wires(modes_out_bra=modes, modes_out_ket=modes)

    def test_rshift(self):
        ket = Coherent([0, 1], 1)
        unitary = Dgate([0], 1)
        u_component = CircuitComponent.from_attributes(
            unitary.name, unitary.representation, unitary.wires
        )
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent.from_attributes(
            channel.name, channel.representation, channel.wires
        )

        dm = ket >> channel
        assert isinstance(dm, DM)

        assert isinstance(dm >> unitary >> channel, DM)
        assert isinstance(dm >> channel >> unitary, DM)
        assert isinstance(dm >> u_component, CircuitComponent)
        assert isinstance(dm >> ch_component, CircuitComponent)

    def test_repr(self):
        ket = Coherent([0, 1], 1)
        channel = Attenuator([1], 1)
        dm = ket >> channel
        dm_component = CircuitComponent.from_attributes(dm.name, dm.representation, dm.wires)

        assert repr(dm) == "DM(name=None, modes=[0, 1])"
        assert repr(dm_component) == "CircuitComponent(name=None, modes=[0, 1])"


class TestCoherent:
    r"""
    Tests for the ``Coherent`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        state = Coherent(modes, x, y)

        assert state.name == "Coherent"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``x``"):
            Coherent(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``y``"):
            Coherent(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = Coherent([0], 1, 1)
        state2 = Coherent([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = Coherent([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.x.value = 3

        state2.x.value = 2
        assert state2.x.value == 2

        state3.y.value = 2
        assert state3.y.value == 2

    def test_representation(self):
        rep1 = Coherent(modes=[0], x=0.1, y=0.2).representation
        assert math.allclose(rep1.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep1.b, [[0.1 + 0.2j]])
        assert math.allclose(rep1.c, [0.97530991])

        rep2 = Coherent(modes=[0, 1], x=0.1, y=[0.2, 0.3]).representation
        assert math.allclose(rep2.A, np.zeros((1, 2, 2)))
        assert math.allclose(rep2.b, [[0.1 + 0.2j, 0.1 + 0.3j]])
        assert math.allclose(rep2.c, [0.9277434863])

        rep3 = Coherent(modes=[1], x=0.1).representation
        assert math.allclose(rep3.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep3.b, [[0.1]])
        assert math.allclose(rep3.c, [0.9950124791926823])

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Coherent(modes=[0], x=[0.1, 0.2]).representation


class TestNumber:
    r"""
    Tests for the ``Number`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    n = [[1], 1, [1, 2]]
    cutoff = [None, 3, 4]

    @pytest.mark.parametrize("modes,n,cutoff", zip(modes, n, cutoff))
    def test_init(self, modes, n, cutoff):
        state = Number(modes, n, cutoff)

        assert state.name == "N"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``n``"):
            Number(modes=[0, 1], n=[2, 3, 4])

        with pytest.raises(ValueError, match="The number of photons per mode"):
            Number(modes=[0, 1], n=3, cutoff=2)

    def test_representation(self):
        rep1 = Number(modes=[0, 1], n=[2, 3], cutoff=4).representation
        assert math.allclose(rep1.array, [[0, 0, 1, 0], [0, 0, 0, 1]])

        rep2 = Number(modes=[0, 1], n=[2, 3]).representation
        assert rep2.array.shape == (2, 100)
        assert rep2.array[0, 2] == 1
        assert rep2.array[1, 3] == 1

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Coherent(modes=[0], x=[0.1, 0.2]).representation


class TestVacuum:
    r"""
    Tests for the ``Vacuum`` class.
    """

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_init(self, modes):
        state = Vacuum(modes)

        assert state.name == "Vac"
        assert state.modes == sorted(modes)

    @pytest.mark.parametrize("n_modes", [1, 3])
    def test_representation(self, n_modes):
        rep = Vacuum(range(n_modes)).representation

        assert math.allclose(rep.A, np.zeros((1, n_modes, n_modes)))
        assert math.allclose(rep.b, np.zeros((1, n_modes)))
        assert math.allclose(rep.c, [1.0])
