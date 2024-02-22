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

import numpy as np

from mrmustard.physics.representations import Bargmann
from mrmustard.lab_dev.circuit_components import (
    connect,
    add_bra,
    CircuitComponent,
    AdjointView,
    DualView,
)
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Dgate, Attenuator
from mrmustard.lab_dev.wires import Wires

        # assert isinstance(vac >> d0, Ket)
        # assert isinstance(vac >> d01, Ket)
        # assert isinstance(vac >> d012, CircuitComponent)
        # assert isinstance(vac >> a0, DM)
        # assert isinstance(vac >> a0 >> d012, CircuitComponent)
        # assert isinstance(d0 >> d0, Unitary)
        # assert isinstance(d0 >> d01, Unitary)
        # assert isinstance(d0 >> d012, Unitary)
        # assert isinstance(a0 >> a0, Channel)
        # assert isinstance(a0 >> d0, Channel)
        # assert isinstance(d0 >> a0, Channel)
        # assert isinstance(d012 >> a0, Channel)


class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    def test_light_copy(self):
        r"""
        Tests the ``light_copy`` method.
        """
        d = Dgate(modes=[0], x=1, y=2, y_trainable=True)
        d_copy = d.light_copy()

        assert d.x is d_copy.x
        assert d.y is d_copy.y
        assert d.wires is not d_copy.wires

    def test_matmul_one_mode(self):
        r"""
        Tests that ``__matmul__`` produces the correct outputs for one-mode components.
        """
        vac0 = Vacuum([0])
        d0 = Dgate(modes=[0], x=1)
        a0 = Attenuator(modes=[0], transmissivity=0.9)

        result1 = vac0 @ d0
        result1 = (result1 @ result1.adjoint) @ a0

        assert result1.wires == Wires(modes_out_bra=[0], modes_out_ket=[0])
        assert np.allclose(result1.representation.A, 0)
        assert np.allclose(result1.representation.b, [0.9486833, 0.9486833])
        assert np.allclose(result1.representation.c, 0.40656966)

        result2 = result1 @ vac0.dual @ vac0.dual.adjoint
        assert not result2.wires
        assert np.allclose(result2.representation.A, 0)
        assert np.allclose(result2.representation.b, 0)
        assert np.allclose(result2.representation.c, 0.40656966)

    def test_matmul_multi_modes(self):
        r"""
        Tests that ``__matmul__`` produces the correct outputs for multi-mode components.
        """
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate(modes=[0], x=0.1, y=0.1)
        d1 = Dgate(modes=[1], x=0.1, y=0.1)
        d2 = Dgate(modes=[2], x=0.1, y=0.1)
        a0 = Attenuator(modes=[0], transmissivity=0.8)
        a1 = Attenuator(modes=[1], transmissivity=0.8)
        a2 = Attenuator(modes=[2], transmissivity=0.7)

        result = vac012 @ d0 @ d1 @ d2
        result = result @ result.adjoint @ a0 @ a1 @ a2

        assert result.wires == Wires(modes_out_bra=[0, 1, 2], modes_out_ket=[0, 1, 2])
        assert np.allclose(result.representation.A, 0)
        assert np.allclose(
            result.representation.b,
            [
                0.08944272 - 0.08944272j,
                0.08944272 - 0.08944272j,
                0.083666 - 0.083666j,
                0.08944272 + 0.08944272j,
                0.08944272 + 0.08944272j,
                0.083666 + 0.083666j,
            ],
        )
        assert np.allclose(result.representation.c, 0.95504196)

    def test_adjoint(self):
        r"""
        Tests the ``adjoint`` method.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2)
        d1_adj = d1.adjoint

        assert isinstance(d1_adj, AdjointView)
        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert d1_adj.representation == d1.representation.conj()

        d1_adj_adj = d1_adj.adjoint
        assert isinstance(d1_adj_adj, CircuitComponent)
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.representation == d1.representation

    def test_dual(self):
        r"""
        Tests the ``dual`` method.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2)
        d1_dual = d1.dual

        assert isinstance(d1_dual, DualView)
        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.representation == d1.representation.conj()

        d1_dual_dual = d1_dual.dual
        assert isinstance(d1_dual_dual, CircuitComponent)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation

    def test_matmul_is_associative(self):
        r"""
        Tests that ``__matmul__`` is associative, meaning ``a @ (b @ c) == (a @ b) @ c``.
        """
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate(modes=[0], x=0.1, y=0.1)
        d1 = Dgate(modes=[1], x=0.1, y=0.1)
        d2 = Dgate(modes=[2], x=0.1, y=0.1)
        a0 = Attenuator(modes=[0], transmissivity=0.8)
        a1 = Attenuator(modes=[1], transmissivity=0.8)
        a2 = Attenuator(modes=[2], transmissivity=0.7)

        result1 = vac012 @ d0 @ d1 @ a0 @ a1 @ a2 @ d2
        result2 = (vac012 @ d0) @ (d1 @ a0) @ a1 @ (a2 @ d2)
        result3 = vac012 @ (d0 @ (d1 @ a0 @ a1) @ a2 @ d2)
        result4 = vac012 @ (d0 @ (d1 @ (a0 @ (a1 @ (a2 @ d2)))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4


class TestAdjointView:
    r"""
    Tests ``AdjointView`` objects.
    """

    def test_init(self):
        r"""
        Tests the ``__init__`` method.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2)
        d1_adj = AdjointView(d1)

        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert d1_adj.representation == d1.representation.conj()

        d1_adj_adj = d1_adj.adjoint
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.representation == d1.representation

    def test_parameters_point_to_original_parameters(self):
        r"""
        Tests that the parameters of an AdjointView object point to those of the original object.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2, x_trainable=True)
        d1_adj = AdjointView(d1)

        d1.x.value = 0.8

        assert d1_adj.x.value == 0.8
        assert d1_adj.representation == d1.representation.conj()


class TestDualView:
    r"""
    Tests ``DualView`` objects.
    """

    def test_init(self):
        r"""
        Tests the ``__init__`` method.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2)
        d1_dual = DualView(d1)

        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.representation == d1.representation.conj()

        d1_dual_dual = DualView(d1_dual)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation

    def test_parameters_point_to_original_parameters(self):
        r"""
        Tests that the parameters of a DualView object point to those of the original object.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2, x_trainable=True)
        d1_dual = DualView(d1)

        d1.x.value = 0.8

        assert d1_dual.x.value == 0.8
        assert d1_dual.representation == d1.representation.conj()


class TestConnect:
    r"""
    Tests the `connect` function.
    """

    def test_ket_only(self):
        r"""
        Tests the ``connect`` function with ket-only components.
        """
        vacuum = Vacuum([0, 1, 2])
        d1 = Dgate(modes=[0, 8, 9], x=1)
        d2 = Dgate(modes=[0, 1, 2], x=1)

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
        d1 = Dgate(modes=[0, 8, 9], x=1)
        d1_adj = d1.adjoint
        a1 = Attenuator(modes=[8], transmissivity=0.1)

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
        vacuum = Vacuum([0, 1, 2])
        d1 = Dgate(modes=[0, 8, 9], x=1)

        components = add_bra([vacuum, d1])

        assert len(components) == 2
        assert components[0] == vacuum @ vacuum.adjoint
        assert components[1] == d1 @ d1.adjoint

    def test_ket_and_bra(self):
        r"""
        Tests the ``add_bra`` function with components with kets and bras.
        """
        vacuum = Vacuum([0, 1, 2])
        a1 = Attenuator(modes=[0, 8, 9], transmissivity=1)

        components = add_bra([vacuum, a1])

        assert len(components) == 2
        assert components[0] == vacuum @ vacuum.adjoint
        assert components[1] == a1
