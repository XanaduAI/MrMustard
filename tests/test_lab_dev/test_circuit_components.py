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

"""Tests for circuit components."""

# pylint: disable=fixme, missing-function-docstring, protected-access, pointless-statement

from unittest.mock import patch

import numpy as np
from ipywidgets import Box, VBox, HBox, HTML
import pytest

from mrmustard import math, settings
from mrmustard.math.parameters import Constant, Variable
from mrmustard.physics.triples import displacement_gate_Abc
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.states import (
    Ket,
    DM,
    Number,
    Vacuum,
    DisplacedSqueezed,
    Coherent,
    SqueezedVacuum,
)
from mrmustard.lab_dev.transformations import Dgate, Attenuator, Unitary, Sgate, Channel
from mrmustard.lab_dev.wires import Wires
from ..random import Abc_triple


# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF


# pylint: disable=too-many-public-methods
class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_init(self, x, y):
        name = "my_component"
        representation = Bargmann(*displacement_gate_Abc(x, y))
        cc = CircuitComponent(representation, wires=[(), (), (1, 8), (1, 8)], name=name)

        assert cc.name == name
        assert list(cc.modes) == [1, 8]
        assert cc.wires == Wires(modes_out_ket={1, 8}, modes_in_ket={1, 8})
        assert cc.representation == representation
        assert cc.manual_shape == [None] * 4

    def test_missing_name(self):
        cc = CircuitComponent(
            Bargmann(*displacement_gate_Abc(0.1, 0.2)), wires=[(), (), (1, 8), (1, 8)]
        )
        cc._name = None
        assert cc.name == "CC18"

    def test_from_bargmann(self):
        cc = CircuitComponent.from_bargmann(displacement_gate_Abc(0.1, 0.2), {}, {}, {0}, {0})
        assert cc.representation == Bargmann(*displacement_gate_Abc(0.1, 0.2))

    def test_modes_init_out_of_order(self):
        m1 = (8, 1)
        m2 = (1, 8)

        r1 = Bargmann(*displacement_gate_Abc(x=[0.1, 0.2]))
        r2 = Bargmann(*displacement_gate_Abc(x=[0.2, 0.1]))

        cc1 = CircuitComponent(r1, wires=[(), (), m1, m1])
        cc2 = CircuitComponent(r2, wires=[(), (), m2, m2])
        assert cc1 == cc2

        r3 = (cc1.adjoint @ cc1).representation
        cc3 = CircuitComponent(r3, wires=[m2, m2, m2, m1])
        cc4 = CircuitComponent(r3, wires=[m2, m2, m2, m2])
        assert cc3.representation == cc4.representation.reorder([0, 1, 2, 3, 4, 5, 7, 6])

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_from_attributes(self, x, y):
        cc = Dgate([1, 8], x=x, y=y)

        cc1 = Dgate._from_attributes(cc.representation, cc.wires, cc.name)
        cc2 = Unitary._from_attributes(cc.representation, cc.wires, cc.name)
        cc3 = CircuitComponent._from_attributes(cc.representation, cc.wires, cc.name)

        assert cc1 == cc
        assert cc2 == cc
        assert cc3 == cc

        assert isinstance(cc1, Unitary) and not isinstance(cc2, Dgate)
        assert isinstance(cc2, Unitary) and not isinstance(cc2, Dgate)
        assert isinstance(cc3, CircuitComponent) and not isinstance(cc3, Unitary)

    def test_from_to_quadrature(self):
        c = Dgate([0], x=0.1, y=0.2) >> Sgate([0], r=1.0, phi=0.1)
        cc = CircuitComponent._from_attributes(c.representation, c.wires, c.name)
        ccc = CircuitComponent.from_quadrature(tuple(), tuple(), (0,), (0,), cc.quadrature())
        assert cc == ccc

    def test_adjoint(self):
        d1 = Dgate([1, 8], x=0.1, y=0.2)
        d1_adj = d1.adjoint

        assert isinstance(d1_adj, CircuitComponent)
        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert (
            d1_adj.representation == d1.representation.conj()
        )  # this holds for the Dgate but not in general

        d1_adj_adj = d1_adj.adjoint
        assert isinstance(d1_adj_adj, CircuitComponent)
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.representation == d1.representation

    def test_dual(self):
        d1 = Dgate([1, 8], x=0.1, y=0.2)
        d1_dual = d1.dual
        vac = Vacuum([1, 8])

        assert isinstance(d1_dual, CircuitComponent)
        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert (vac >> d1 >> d1_dual).representation == vac.representation
        assert (vac >> d1_dual >> d1).representation == vac.representation

        d1_dual_dual = d1_dual.dual
        assert isinstance(d1_dual_dual, CircuitComponent)
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.representation == d1.representation

    def test_light_copy(self):
        d1 = CircuitComponent(
            Bargmann(*displacement_gate_Abc(0.1, 0.1)), wires=[(), (), (1,), (1,)]
        )
        d1_cp = d1._light_copy()

        assert d1_cp.parameter_set is d1.parameter_set
        assert d1_cp.representation is d1.representation
        assert d1_cp.wires is not d1.wires

    def test_on(self):
        assert Vacuum([1, 2]).on([3, 4]).modes == [3, 4]
        assert Number([3], n=4).on([9]).modes == [9]

        d89 = DisplacedSqueezed([8, 9], x=[1, 2], y=3, r_trainable=True)
        d67 = d89.on([6, 7])
        assert isinstance(d67.x, Constant)
        assert math.allclose(d89.x.value, d67.x.value)
        assert isinstance(d67.y, Constant)
        assert math.allclose(d89.y.value, d67.y.value)
        assert isinstance(d67.r, Variable)
        assert math.allclose(d89.r.value, d67.r.value)
        assert bool(d67.parameter_set) is True
        assert d67._representation is d89._representation

    def test_on_error(self):
        with pytest.raises(ValueError):
            Vacuum([1, 2]).on([3])

    def test_to_fock_ket(self):
        vac = Vacuum([1, 2])
        vac_fock = vac.to_fock(shape=[1, 2])
        assert vac_fock.representation == Fock(np.array([[1], [0]]))

    def test_to_fock_Number(self):
        num = Number([3], n=4)
        num_f = num.to_fock(shape=(6,))
        assert num_f.representation == Fock(np.array([0, 0, 0, 0, 1, 0]))

    def test_to_fock_Dgate(self):
        d = Dgate([1], x=0.1, y=0.1)
        d_fock = d.to_fock(shape=(4, 6))
        assert d_fock.representation == Fock(
            math.hermite_renormalized(*displacement_gate_Abc(x=0.1, y=0.1), shape=(4, 6))
        )

    def test_to_fock_poly_exp(self):
        A, b, _ = Abc_triple(3)
        c = np.random.random((1, 5))
        barg = Bargmann(A, b, c)
        cc = CircuitComponent(barg, wires=[(), (), (0, 1), ()]).to_fock(shape=(10, 10))
        poly = math.hermite_renormalized(A, b, 1, (10, 10, 5))
        assert np.allclose(cc.representation.data, np.einsum("ijk,k", poly, c[0]))

    def test_add(self):
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([1], x=0.2, y=0.2)

        d12 = d1 + d2
        assert d12.representation == d1.representation + d2.representation

    def test_sub(self):
        s1 = DisplacedSqueezed([1], x=1.0, y=0.5, r=0.1)
        s2 = DisplacedSqueezed([1], x=0.5, y=0.2, r=0.2)
        s12 = s1 - s2
        assert s12.representation == s1.representation - s2.representation

    def test_mul(self):
        d1 = Dgate([1], x=0.1, y=0.1)

        assert (d1 * 3).representation == d1.representation * 3
        assert (3 * d1).representation == d1.representation * 3
        assert isinstance(d1 * 3, Unitary)

    def test_truediv(self):
        d1 = Dgate([1], x=0.1, y=0.1)

        assert (d1 / 3).representation == d1.representation / 3
        assert isinstance(d1 / 3, Unitary)

    def test_add_error(self):
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.2, y=0.2)

        with pytest.raises(ValueError):
            d1 + d2

    def test_eq(self):
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)

        assert d1 == d1._light_copy()
        assert d1 != d2

    def test_matmul(self):
        vac012 = Vacuum([0, 1, 2])
        d012 = Dgate([0, 1, 2], x=0.1, y=0.1)
        a0 = Attenuator([0], 0.8)
        a1 = Attenuator([1], 0.8)
        a2 = Attenuator([2], 0.7)

        result = vac012 @ d012
        result = result @ result.adjoint @ a0 @ a1 @ a2

        assert result.wires == Wires(modes_out_bra={0, 1, 2}, modes_out_ket={0, 1, 2})
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

    def test_matmul_scalar(self):
        d0 = Dgate([0], x=0.1, y=0.1)
        result = d0 @ 0.8
        assert math.allclose(result.representation.A, d0.representation.A)
        assert math.allclose(result.representation.b, d0.representation.b)
        assert math.allclose(result.representation.c, 0.8 * d0.representation.c)
        result2 = 0.8 @ d0
        assert math.allclose(result2.representation.A, d0.representation.A)
        assert math.allclose(result2.representation.b, d0.representation.b)
        assert math.allclose(result2.representation.c, 0.8 * d0.representation.c)

    def test_rshift_all_bargmann(self):
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

        result = vac012 >> d0 >> d1 >> d2 >> a0 >> a1 >> a2

        assert result.wires == Wires(modes_out_bra={0, 1, 2}, modes_out_ket={0, 1, 2})
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
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.1)
        a0 = Attenuator([0], transmissivity=0.8)
        a1 = Attenuator([1], transmissivity=0.8)
        a2 = Attenuator([2], transmissivity=0.7)

        N = 10
        r1 = (vac012 >> d0 >> d1 >> d2 >> a0 >> a1 >> a2).to_fock(N)
        r2 = (
            vac012.to_fock(N)
            >> d0.to_fock(N)
            >> d1.to_fock(N)
            >> d2.to_fock(N)
            >> a0.to_fock(N)
            >> a1.to_fock(N)
            >> a2.to_fock(N)
        )

        assert r1 == r2

    @pytest.mark.parametrize("shape", [5, 6])
    def test_rshift_bargmann_and_fock(self, shape):
        settings.AUTOSHAPE_MAX = shape
        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.4, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.5)
        a1 = Attenuator([1], transmissivity=0.9)
        n12 = Number([1, 2], n=1).dual

        # bargmann >> fock
        r1 = vac12 >> d1 >> d2 >> a1 >> n12

        # fock >> bargmann
        r2 = vac12.to_fock(shape) >> d1 >> d2 >> a1 >> n12

        # bargmann >> fock >> bargmann
        r3 = vac12 >> d1.to_fock(shape) >> d2 >> a1 >> n12

        assert np.allclose(r1, r2)
        assert np.allclose(r1, r3)

        settings.AUTOSHAPE_MAX = 50

    def test_rshift_error(self):
        vac012 = Vacuum([0, 1, 2])
        d0 = Dgate([0], x=0.1, y=0.1)
        d0._wires = Wires()

        with pytest.raises(ValueError, match="not clear"):
            vac012 >> d0

    def test_rshift_ketbra_with_ket(self):
        a1 = Attenuator([1], transmissivity=0.8)
        n1 = Number([1, 2], n=1).dual

        assert a1 >> n1 == a1 @ n1 @ n1.adjoint

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

    def test_rshift_scalar(self):
        d0 = Dgate([0], x=0.1, y=0.1)
        result = 0.8 >> d0
        assert math.allclose(result, 0.8 * d0.representation.c)

        result2 = d0 >> 0.8
        assert math.allclose(result2.representation.c, 0.8 * d0.representation.c)

    def test_repr(self):
        c1 = CircuitComponent(wires=Wires(modes_out_ket=(0, 1, 2)))
        c2 = CircuitComponent(wires=Wires(modes_out_ket=(0, 1, 2)), name="my_component")

        assert repr(c1) == "CircuitComponent(modes=[0, 1, 2], name=CC012)"
        assert repr(c2) == "CircuitComponent(modes=[0, 1, 2], name=my_component)"

    def test_to_fock_keeps_bargmann(self):
        "tests that to_fock doesn't lose the bargmann representation"
        coh = Coherent([0], x=1.0)
        coh.to_fock(20)
        assert coh.bargmann_triple() == Coherent([0], x=1.0).bargmann_triple()

    def test_fock_component_no_bargmann(self):
        "tests that a fock component doesn't have a bargmann representation by default"
        coh = Coherent([0], x=1.0)
        CC = Ket.from_fock([0], coh.fock(20), batched=False)
        with pytest.raises(AttributeError):
            CC.bargmann_triple()  # pylint: disable=pointless-statement

    def test_quadrature_ket(self):
        "tests that transforming to quadrature and back gives the same ket"
        ket = SqueezedVacuum([0], 0.4, 0.5) >> Dgate([0], 0.3, 0.2)
        back = Ket.from_quadrature([0], ket.quadrature())
        assert ket == back

    def test_quadrature_dm(self):
        "tests that transforming to quadrature and back gives the same density matrix"
        dm = SqueezedVacuum([0], 0.4, 0.5) >> Dgate([0], 0.3, 0.2) >> Attenuator([0], 0.9)
        back = DM.from_quadrature([0], dm.quadrature())
        assert dm == back

    def test_quadrature_unitary(self):
        U = Sgate([0], 0.5, 0.4) >> Dgate([0], 0.3, 0.2)
        back = Unitary.from_quadrature([0], [0], U.quadrature())
        assert U == back

    def test_quadrature_channel(self):
        C = Sgate([0], 0.5, 0.4) >> Dgate([0], 0.3, 0.2) >> Attenuator([0], 0.9)
        back = Channel.from_quadrature([0], [0], C.quadrature())
        assert C == back

    @pytest.mark.parametrize("is_fock,widget_cls", [(False, Box), (True, HBox)])
    @patch("mrmustard.lab_dev.circuit_components.display")
    def test_ipython_repr(self, mock_display, is_fock, widget_cls):
        """Test the IPython repr function."""
        dgate = Dgate([1], x=0.1, y=0.1)
        if is_fock:
            dgate = dgate.to_fock()
        dgate._ipython_display_()  # pylint:disable=protected-access
        [box] = mock_display.call_args.args
        assert isinstance(box, Box)
        [wires_widget, rep_widget] = box.children
        assert isinstance(wires_widget, HTML)
        assert type(rep_widget) is widget_cls

    @patch("mrmustard.lab_dev.circuit_components.display")
    def test_ipython_repr_invalid_obj(self, mock_display):
        """Test the IPython repr function."""
        dgate = Dgate([1, 2], x=0.1, y=0.1).to_fock()
        dgate._ipython_display_()  # pylint:disable=protected-access
        [box] = mock_display.call_args.args
        assert isinstance(box, VBox)
        [title_widget, wires_widget] = box.children
        assert isinstance(title_widget, HTML)
        assert isinstance(wires_widget, HTML)

    def test_serialize_default_behaviour(self):
        """Test the default serializer."""
        name = "my_component"
        rep = Bargmann(*displacement_gate_Abc(0.1, 0.4))
        cc = CircuitComponent(rep, wires=[(), (), (1, 8), (1, 8)], name=name)
        kwargs, arrays = cc._serialize()
        assert kwargs == {
            "class": f"{CircuitComponent.__module__}.CircuitComponent",
            "wires": cc.wires.sorted_args,
            "rep_class": f"{Bargmann.__module__}.Bargmann",
            "name": name,
        }
        assert arrays == {"A": rep.A, "b": rep.b, "c": rep.c}

    def test_serialize_fail_when_no_modes_input(self):
        """Test that the serializer fails if no modes or name+wires are present."""

        class MyComponent(CircuitComponent):
            """A dummy class without a valid modes kwarg."""

            def __init__(self, rep, custom_modes):
                super().__init__(rep, wires=[custom_modes] * 4, name="my_component")

        cc = MyComponent(Bargmann(*displacement_gate_Abc(0.1, 0.4)), [0, 1])
        with pytest.raises(
            TypeError, match="MyComponent does not seem to have any wires construction method"
        ):
            cc._serialize()
