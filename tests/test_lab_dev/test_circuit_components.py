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

r"""
Tests for circuit components.
"""

from mrmustard.lab_dev.circuit_components import connect, add_bra, CircuitComponent
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Dgate, Attenuator


class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    def test_light_copy(self):
        r"""
        Tests the ``light_copy`` method.
        """
        d = Dgate(x=1, y=2, y_trainable=True)
        d_copy = d.light_copy()

        assert d.x is d_copy.x
        assert d.y is d_copy.y
        assert d.wires is not d_copy.wires


class TestConnect:
    r"""
    Tests the `connect` function.
    """

    def test_ket_only(self):
        r"""
        Tests the ``connect`` function with ket-only components.
        """
        vacuum = Vacuum(3)
        d1 = Dgate(1, modes=[0, 8, 9])
        d2 = Dgate(1, modes=[0, 1, 2])

        components = [vacuum, d1, d1, d2]
        components = connect(components)

        # check that all the modes are still there and no new modes are added
        assert list(components[0].wires.modes) == [0, 1, 2]
        assert list(components[1].wires.modes) == [0, 8, 9]
        assert list(components[2].wires.modes) == [0, 8, 9]
        assert list(components[3].wires.modes) == [0, 1, 2]

        # check connections on mode 0
        assert components[0].wires.output.ket[0].ids == components[1].wires.input.ket[0].ids
        assert components[1].wires.output.ket[0].ids == components[2].wires.input.ket[0].ids
        assert components[2].wires.output.ket[0].ids == components[3].wires.input.ket[0].ids

        # check connections on mode 1
        assert components[0].wires.output.ket[1].ids == components[3].wires.input.ket[1].ids

        # check connections on mode 2
        assert components[0].wires.output.ket[2].ids == components[3].wires.input.ket[2].ids

        # check connections on mode 8
        assert components[1].wires.output.ket[8].ids == components[2].wires.input.ket[8].ids

        # check connections on mode 9
        assert components[1].wires.output.ket[9].ids == components[2].wires.input.ket[9].ids

    def test_ket_and_bra(self):
        r"""
        Tests the ``connect`` function with components with kets and bras.
        """
        d1 = Dgate(1, modes=[0, 8, 9])
        d1_adj = d1.adjoint
        a1 = Attenuator(0.1, modes=[8])

        components = connect([d1, d1_adj, a1])

        # check connections on mode 8
        assert components[0].wires.output.ket[8].ids == components[2].wires.input.ket[8].ids
        assert components[1].wires.output.bra[8].ids == components[2].wires.input.bra[8].ids


class TestAddBra:
    r"""
    Tests the `add_bra` function.
    """

    def test_ket_only(self):
        r"""
        Tests the ``add_bra`` function with ket-only components.
        """
        vacuum = Vacuum(3)
        d1 = Dgate(1, modes=[0, 8, 9])

        components = add_bra([vacuum, d1])

        assert len(components) == 4

        assert isinstance(components[0], Vacuum)
        assert components[0].wires.ket and not components[0].wires.bra
        assert isinstance(components[1], CircuitComponent)
        assert not components[1].wires.ket and components[1].wires.bra

        assert isinstance(components[2], Dgate)
        assert components[2].wires.ket and not components[2].wires.bra
        assert isinstance(components[3], CircuitComponent)
        assert not components[3].wires.ket and components[3].wires.bra

    def test_ket_and_bra(self):
        r"""
        Tests the ``add_bra`` function with components with kets and bras.
        """
        vacuum = Vacuum(3)
        a1 = Attenuator(1, modes=[0, 8, 9])

        components = add_bra([vacuum, a1])

        assert len(components) == 3

        assert isinstance(components[0], Vacuum)
        assert components[0].wires.ket and not components[0].wires.bra
        assert isinstance(components[1], CircuitComponent)
        assert not components[1].wires.ket and components[1].wires.bra

        assert isinstance(components[2], Attenuator)
        assert components[2].wires.ket and components[2].wires.bra
