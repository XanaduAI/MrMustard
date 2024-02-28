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
import pytest

from mrmustard.physics.triples import attenuator_Abc, displacement_gate_Abc, vacuum_state_Abc
from mrmustard.physics.representations import Bargmann
from mrmustard.lab_dev.circuit_components import (
    add_bra,
    CircuitComponent,
    AdjointView,
    DualView,
)
from mrmustard.lab_dev.wires import Wires


class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_init(self, x, y):
        name = "my_component"
        representation = Bargmann(*displacement_gate_Abc(x, y))
        modes = [1, 8]
        cc = CircuitComponent(name, representation, modes_out_ket=modes, modes_in_ket=modes)

        assert cc.name == name
        assert cc.modes == modes
        assert cc.wires == Wires(modes_out_ket=modes, modes_in_ket=modes)
        assert cc.representation == representation

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_from_attributes(self, x, y):
        name = "my_component"
        Abc = displacement_gate_Abc(x, y)
        representation = Bargmann(*Abc)
        modes = [1, 8]

        cc1 = CircuitComponent(name, representation, modes_out_ket=modes, modes_in_ket=modes)
        cc2 = CircuitComponent.from_attributes(cc1.name, cc1.representation, cc1.wires)

        assert cc1 == cc2

    def test_adjoint(self):
        Abc1 = displacement_gate_Abc(x=0.1, y=0.2)
        modes1 = [1, 8]

        d1 = CircuitComponent("d1", Bargmann(*Abc1), modes_out_ket=modes1, modes_in_ket=modes1)
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
        Abc1 = displacement_gate_Abc(x=0.1, y=0.2)
        modes1 = [1, 8]

        d1 = CircuitComponent("d1", Bargmann(*Abc1), modes_out_ket=modes1, modes_in_ket=modes1)
        d1_dual = d1.dual

        assert isinstance(d1_dual, DualView)
        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.representation == d1.representation.conj()

        d1_dual_dual = d1_dual.dual
        assert isinstance(d1_dual_dual, CircuitComponent)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation

    def test_light_copy(self):
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )

        d1_cp = d1.light_copy()

        assert d1_cp.parameter_set is d1.parameter_set
        assert d1_cp.representation is d1.representation
        assert d1_cp.wires is not d1.wires

    def test_eq(self):
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d2 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        assert d1 == d1.light_copy()
        assert not d1 == d2

    def test_matmul(self):
        vac012 = CircuitComponent(
            "",
            Bargmann(*vacuum_state_Abc(3)),
            modes_out_ket=[0, 1, 2],
        )
        d012 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])),
            modes_out_ket=[0, 1, 2],
            modes_in_ket=[0, 1, 2],
        )
        a0 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[0],
            modes_in_bra=[0],
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        a1 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[1],
            modes_in_bra=[1],
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        a2 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.7)),
            modes_out_bra=[2],
            modes_in_bra=[2],
            modes_out_ket=[2],
            modes_in_ket=[2],
        )

        result = vac012 @ d012
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

    def test_matmul_one_mode_Dgate_contraction(self):
        r"""
        Tests that ``__matmul__`` produces the correct outputs for two Dgate with the formula well-known.
        """
        alpha = 1.5 + 0.7888 * 1j
        beta = -0.1555 + 1j * 2.1

        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(x=alpha.real, y=alpha.imag)),
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        d2 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(x=beta.real, y=beta.imag)),
            modes_out_ket=[0],
            modes_in_ket=[0],
        )

        result1 = d2 @ d1
        correct_c = np.exp(-0.5 * (abs(alpha + beta) ** 2)) * np.exp(
            (alpha * np.conj(beta) - np.conj(alpha) * beta) / 2
        )

        assert np.allclose(result1.representation.c, correct_c)

    def test_matmul_is_associative(self):
        d0 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d2 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        a0 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[0],
            modes_in_bra=[0],
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        a1 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[1],
            modes_in_bra=[1],
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        a2 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.7)),
            modes_out_bra=[2],
            modes_in_bra=[2],
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        result1 = d0 @ d1 @ a0 @ a1 @ a2 @ d2
        result2 = d0 @ (d1 @ a0) @ a1 @ (a2 @ d2)
        result3 = d0 @ (d1 @ a0 @ a1) @ a2 @ d2
        result4 = d0 @ (d1 @ (a0 @ (a1 @ (a2 @ d2))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_rshift(self):
        vac012 = CircuitComponent(
            "",
            Bargmann(*vacuum_state_Abc(3)),
            modes_out_ket=[0, 1, 2],
        )
        d0 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d2 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        a0 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[0],
            modes_in_bra=[0],
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        a1 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[1],
            modes_in_bra=[1],
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        a2 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.7)),
            modes_out_bra=[2],
            modes_in_bra=[2],
            modes_out_ket=[2],
            modes_in_ket=[2],
        )

        result = vac012 >> d0 >> d1 >> d2 >> a0 >> a1 >> a2

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

    def test_rshift_is_associative(self):
        vac012 = CircuitComponent(
            "",
            Bargmann(*vacuum_state_Abc(3)),
            modes_out_ket=[0, 1, 2],
        )
        d0 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d2 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        a0 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[0],
            modes_in_bra=[0],
            modes_out_ket=[0],
            modes_in_ket=[0],
        )
        a1 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[1],
            modes_in_bra=[1],
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        a2 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.7)),
            modes_out_bra=[2],
            modes_in_bra=[2],
            modes_out_ket=[2],
            modes_in_ket=[2],
        )
        result1 = vac012 >> d0 >> d1 >> a0 >> a1 >> a2 >> d2
        result2 = (vac012 >> d0) >> (d1 >> a0) >> a1 >> (a2 >> d2)
        result3 = vac012 >> (d0 >> (d1 >> a0 >> a1) >> a2 >> d2)
        result4 = vac012 >> (d0 >> (d1 >> (a0 >> (a1 >> (a2 >> d2)))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_repr(self):
        c1 = CircuitComponent("", modes_out_ket=[0, 1, 2])
        assert repr(c1) == "CircuitComponent(name=None, modes=[0, 1, 2])"


class TestAdjointView:
    r"""
    Tests ``AdjointView`` objects.
    """

    def test_init(self):
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d1_adj = AdjointView(d1)

        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert d1_adj.representation == d1.representation.conj()

        d1_adj_adj = d1_adj.adjoint
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.representation == d1.representation


class TestDualView:
    r"""
    Tests ``DualView`` objects.
    """

    def test_init(self):
        r"""
        Tests the ``__init__`` method.
        """
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        d1_dual = DualView(d1)

        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.representation == d1.representation.conj()

        d1_dual_dual = DualView(d1_dual)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation


class TestAddBra:
    r"""
    Tests the `add_bra` function.
    """

    def test_ket_only(self):
        vac012 = CircuitComponent(
            "",
            Bargmann(*vacuum_state_Abc(3)),
            modes_out_ket=[0, 1, 2],
        )
        d1 = CircuitComponent(
            "",
            Bargmann(*displacement_gate_Abc(0.1, 0.1)),
            modes_out_ket=[1],
            modes_in_ket=[1],
        )
        components = add_bra([vac012, d1])

        assert len(components) == 2
        assert components[0] == vac012 @ vac012.adjoint
        assert components[1] == d1 @ d1.adjoint

    def test_ket_and_bra(self):
        vac012 = CircuitComponent(
            "",
            Bargmann(*vacuum_state_Abc(3)),
            modes_out_ket=[0, 1, 2],
        )
        a1 = CircuitComponent(
            "",
            Bargmann(*attenuator_Abc(0.8)),
            modes_out_bra=[1],
            modes_in_bra=[1],
            modes_out_ket=[1],
            modes_in_ket=[1],
        )

        components = add_bra([vac012, a1])

        assert len(components) == 2
        assert components[0] == vac012 @ vac012.adjoint
        assert components[1] == a1
