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

""" Tests for circuit components. """

# pylint: disable=fixme, missing-function-docstring

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.physics.converters import to_fock
from mrmustard.physics.triples import displacement_gate_Abc
from mrmustard.physics.representations import Bargmann
from mrmustard.lab_dev.circuit_components import CircuitComponent, AdjointView, DualView
from mrmustard.lab_dev.states import Coherent, Number, Vacuum
from mrmustard.lab_dev.transformations import Dgate, Attenuator
from mrmustard.lab_dev.wires import Wires

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF


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
        cc1 = Dgate([1, 8], x=x, y=y)
        cc2 = CircuitComponent.from_attributes(cc1.name, cc1.representation, cc1.wires)

        assert cc1 == cc2

    def test_adjoint(self):
        d1 = Dgate([1, 8], x=0.1, y=0.2)
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
        d1 = Dgate([1, 8], x=0.1, y=0.2)
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

    @pytest.mark.parametrize("shape", [3, [3, 2]])
    def test_to_fock(self, shape):
        vac = Vacuum([1, 2])
        vac_fock = vac.to_fock(shape=shape)
        assert vac_fock.name == vac.name
        assert vac_fock.wires == vac.wires
        assert vac_fock.representation == to_fock(vac.representation, shape)

        n = Number([3], n=4)
        n_fock = n.to_fock(shape=shape)
        assert n_fock.name == n.name
        assert n_fock.wires == n.wires
        assert n_fock.representation == to_fock(n.representation, shape)

        d = Dgate([1], x=0.1, y=0.1)
        d_fock = d.to_fock(shape=shape)
        assert d_fock.name == d.name
        assert d_fock.wires == d.wires
        assert d_fock.representation == to_fock(d.representation, shape)

    def test_eq(self):
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)

        assert d1 == d1.light_copy()
        assert d1 != d2

    def test_matmul(self):
        vac012 = Vacuum([0, 1, 2])
        d012 = Dgate([0, 1, 2], x=0.1, y=0.1)
        a0 = Attenuator([0], 0.8)
        a1 = Attenuator([1], 0.8)
        a2 = Attenuator([2], 0.7)

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

        d1 = Dgate([0], x=alpha.real, y=alpha.imag)
        d2 = Dgate([0], x=beta.real, y=beta.imag)

        result1 = d2 @ d1
        correct_c = np.exp(-0.5 * (abs(alpha + beta) ** 2)) * np.exp(
            (alpha * np.conj(beta) - np.conj(alpha) * beta) / 2
        )

        assert np.allclose(result1.representation.c, correct_c)

    def test_matmul_is_associative(self):
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

        result1 = d0 @ d1 @ a0 @ a1 @ a2 @ d2
        result2 = d0 @ (d1 @ a0) @ a1 @ (a2 @ d2)
        result3 = d0 @ (d1 @ a0 @ a1) @ a2 @ d2
        result4 = d0 @ (d1 @ (a0 @ (a1 @ (a2 @ d2))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_lshift(self):
        ket01 = Coherent([0, 1], 1)
        n0 = Number([0], n=5)
        unitary = Dgate([0], 1)
        channel = Attenuator([1], 1)

        assert ket01 << n0 == ket01 >> n0.dual
        assert ket01 >> unitary << n0 == ket01 >> unitary >> n0.dual
        assert ket01 >> channel << n0 == ket01 >> channel >> n0.dual

    def test_rshift_all_bargmann(self):
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

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

    def test_rshift_all_fock(self):
        settings.AUTOCUTOFF_MAX_CUTOFF = 10

        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

        r1 = (vac012 >> d0 >> d1 >> d2 >> a0 >> a1 >> a2).to_fock()
        r2 = (
            vac012.to_fock()
            >> d0.to_fock()
            >> d1.to_fock()
            >> d2.to_fock()
            >> a0
            >> a1.to_fock().to_fock()
            >> a2.to_fock()
        )

        assert r1 == r2

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0

    @pytest.mark.parametrize("autocutoff", [5, 6])
    def test_rshift_bargmann_and_fock(self, autocutoff):
        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff

        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.2)
        d12 = Dgate([1, 2], x=0.1, y=[0.1, 0.2])
        a1 = Attenuator([1], transmissivity=0.8)
        n12 = Number([1, 2], n=1)

        # bargmann >> fock
        r1 = vac12 >> d1 >> d2 >> a1 << n12

        # fock >> bargmann
        r2 = vac12.to_fock() >> d1 >> d2 >> a1 << n12

        # bargmann >> fock >> bargmann
        r3 = vac12 >> d1.to_fock() >> d2 >> a1 << n12

        # fock only
        r4 = vac12.to_fock() >> d12.to_fock() >> a1.to_fock() << n12.to_fock()

        assert math.allclose(r1.representation.array, r2.representation.array)
        assert math.allclose(r1.representation.array, r3.representation.array)
        assert math.allclose(r1.representation.array, r4.representation.array)

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0

    def test_rshift_is_associative(self):
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

        result1 = vac012 >> d0 >> d1 >> a0 >> a1 >> a2 >> d2
        result2 = (vac012 >> d0) >> (d1 >> a0) >> a1 >> (a2 >> d2)
        result3 = vac012 >> (d0 >> (d1 >> a0 >> a1) >> a2 >> d2)
        result4 = vac012 >> (d0 >> (d1 >> (a0 >> (a1 >> (a2 >> d2)))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_repr(self):
        c1 = CircuitComponent("", modes_out_ket=[0, 1, 2])
        c2 = CircuitComponent("my_component", modes_out_ket=[0, 1, 2])

        assert repr(c1) == "CircuitComponent(name=None, modes=[0, 1, 2])"
        assert repr(c2) == "CircuitComponent(name=my_component, modes=[0, 1, 2])"


class TestAdjointView:
    r"""
    Tests ``AdjointView`` objects.
    """

    def test_init(self):
        d1 = Dgate([1], x=0.1, y=0.1)
        d1_adj = AdjointView(d1)

        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert d1_adj.representation == d1.representation.conj()

        d1_adj_adj = d1_adj.adjoint
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.representation == d1.representation

    def test_repr(self):
        c1 = CircuitComponent("", modes_out_ket=[0, 1, 2])
        c2 = CircuitComponent("my_component", modes_out_ket=[0, 1, 2])

        assert repr(c1.adjoint) == "CircuitComponent(name=None, modes=[0, 1, 2])"
        assert repr(c2.adjoint) == "CircuitComponent(name=my_component, modes=[0, 1, 2])"

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
        d1 = Dgate([1], x=0.1, y=0.1)
        d1_dual = DualView(d1)

        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.representation == d1.representation.conj()

        d1_dual_dual = DualView(d1_dual)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation

    def test_repr(self):
        c1 = CircuitComponent("", modes_out_ket=[0, 1, 2])
        c2 = CircuitComponent("my_component", modes_out_ket=[0, 1, 2])

        assert repr(c1.dual) == "CircuitComponent(name=None, modes=[0, 1, 2])"
        assert repr(c2.dual) == "CircuitComponent(name=my_component, modes=[0, 1, 2])"

    def test_parameters_point_to_original_parameters(self):
        r"""
        Tests that the parameters of a DualView object point to those of the original object.
        """
        d1 = Dgate(modes=[0], x=0.1, y=0.2, x_trainable=True)
        d1_dual = DualView(d1)

        d1.x.value = 0.8

        assert d1_dual.x.value == 0.8
        assert d1_dual.representation == d1.representation.conj()
