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

from unittest.mock import patch

import numpy as np
import pytest
from ipywidgets import HTML, Box, HBox, VBox

from mrmustard import math, settings
from mrmustard.lab import (
    DM,
    Attenuator,
    BSgate,
    Channel,
    Circuit,
    CircuitComponent,
    Coherent,
    Dgate,
    DisplacedSqueezed,
    Interferometer,
    Ket,
    Map,
    Number,
    Operation,
    Sgate,
    SqueezedVacuum,
    Unitary,
    Vacuum,
)
from mrmustard.lab.circuit_components import ReprEnum
from mrmustard.math.parameters import Constant, Variable
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.triples import displacement_gate_Abc, identity_Abc
from mrmustard.physics.wires import Wires
from mrmustard.training import Optimizer

from ..random import Abc_triple


class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_init(self, x, y):
        name = "my_component"
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(x, y))
        wires = Wires(set(), set(), {1, 8}, {1, 8})
        cc = CircuitComponent(ansatz=ansatz, wires=wires, name=name)

        assert cc.name == name
        assert cc.modes == (1, 8)
        assert cc.wires == Wires(modes_out_ket={1, 8}, modes_in_ket={1, 8})
        assert cc.ansatz == ansatz
        assert cc.manual_shape == (None,) * 4

    def test_missing_name(self):
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(0.1, 0.2))
        wires = Wires(set(), set(), {1, 8}, {1, 8})
        cc = CircuitComponent(ansatz=ansatz, wires=wires)
        cc._name = None
        assert cc.name == "CC18"

    def test_from_bargmann(self):
        cc = CircuitComponent.from_bargmann(displacement_gate_Abc(0.1, 0.2), {}, {}, {0}, {0})
        assert cc.ansatz == PolyExpAnsatz(*displacement_gate_Abc(0.1, 0.2))

    def test_from_attributes(self):
        cc = Dgate(1, x=0.1, y=0.2)

        cc1 = Dgate._from_attributes(cc.ansatz, cc.wires, cc.name)
        cc2 = Unitary._from_attributes(cc.ansatz, cc.wires, cc.name)
        cc3 = CircuitComponent._from_attributes(cc.ansatz, cc.wires, cc.name)

        assert cc1 == cc
        assert cc2 == cc
        assert cc3 == cc

        assert isinstance(cc1, Unitary) and not isinstance(cc2, Dgate)
        assert isinstance(cc2, Unitary) and not isinstance(cc2, Dgate)
        assert isinstance(cc3, CircuitComponent) and not isinstance(cc3, Unitary)

    def test_from_to_quadrature(self):
        c = Dgate(0, x=0.1, y=0.2) >> Sgate(0, r=1.0, phi=0.1)
        cc = CircuitComponent(ansatz=c.ansatz, wires=c.wires, name=c.name)
        ccc = CircuitComponent.from_quadrature((), (), (0,), (0,), cc.quadrature_triple())
        assert cc == ccc

    def test_adjoint(self):
        d1 = Dgate(1, x=0.1, y=0.2)
        d1_adj = d1.adjoint

        assert isinstance(d1_adj, CircuitComponent)
        assert d1_adj.name == d1.name
        assert d1_adj.wires == d1.wires.adjoint
        assert d1_adj.parameters == d1.parameters
        assert d1_adj.ansatz == d1.ansatz.conj  # this holds for the Dgate but not in general

        d1_adj_adj = d1_adj.adjoint
        assert isinstance(d1_adj_adj, CircuitComponent)
        assert d1_adj_adj.wires == d1.wires
        assert d1_adj_adj.parameters == d1_adj.parameters
        assert d1_adj_adj.parameters == d1.parameters
        assert d1_adj_adj.ansatz == d1.ansatz

    def test_dual(self):
        d1 = Dgate(1, x=0.1, y=0.2)
        d1_dual = d1.dual
        vac = Vacuum(1)

        assert isinstance(d1_dual, CircuitComponent)
        assert d1_dual.name == d1.name
        assert d1_dual.wires == d1.wires.dual
        assert d1_dual.parameters == d1.parameters
        assert (vac >> d1 >> d1_dual).ansatz == vac.ansatz
        assert (vac >> d1_dual >> d1).ansatz == vac.ansatz

        d1_dual_dual = d1_dual.dual
        assert isinstance(d1_dual_dual, CircuitComponent)
        assert d1_dual_dual.parameters == d1_dual.parameters
        assert d1_dual_dual.parameters == d1.parameters
        assert d1_dual_dual.wires == d1.wires
        assert d1_dual_dual.ansatz == d1.ansatz

    def test_light_copy(self):
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(0.1, 0.1))
        wires = Wires(set(), set(), {1}, {1})
        d1 = CircuitComponent(ansatz=ansatz, wires=wires)
        d1_cp = d1._light_copy()

        assert d1_cp.parameters is d1.parameters
        assert d1_cp.ansatz is d1.ansatz
        assert d1_cp.wires is not d1.wires

    def test_on(self):
        assert Vacuum([1, 2]).on([3, 4]).modes == (3, 4)
        assert Number(3, n=4).on(9).modes == (9,)

        d89 = DisplacedSqueezed(8, x=1, y=3, r_trainable=True)
        d67 = d89.on(6)
        assert isinstance(d67.parameters.x, Constant)
        assert math.allclose(d89.parameters.x.value, d67.parameters.x.value)
        assert isinstance(d67.parameters.y, Constant)
        assert math.allclose(d89.parameters.y.value, d67.parameters.y.value)
        assert isinstance(d67.parameters.r, Variable)
        assert math.allclose(d89.parameters.r.value, d67.parameters.r.value)
        assert bool(d67.parameters) is True
        assert d67.ansatz is d89.ansatz

    def test_on_error(self):
        with pytest.raises(ValueError):
            Vacuum((1, 2)).on(3)

    def test_to_fock_ket(self):
        vac = Vacuum((1, 2))
        vac_fock = vac.to_fock(shape=(1, 2))
        assert vac_fock.ansatz == ArrayAnsatz(np.array([[1], [0]]))

    def test_to_fock_bargmann_Number(self):
        num = Number(3, n=4)
        num_f = num.to_fock(shape=(6,))
        assert num_f.ansatz == ArrayAnsatz(np.array([0, 0, 0, 0, 1, 0]))

        num_barg = num_f.to_bargmann()
        A_exp, b_exp, _ = identity_Abc(1)
        assert math.allclose(num_barg.ansatz.A, A_exp)
        assert math.allclose(num_barg.ansatz.b, b_exp)
        assert math.allclose(num_barg.ansatz.c, num_f.ansatz.array)

    def test_to_fock_bargmann_Dgate(self):
        d = Dgate(1, x=0.1, y=0.1)
        d_barg = d.to_bargmann()
        assert d is d_barg

        d_fock = d.to_fock(shape=(4, 6))
        assert d_fock.ansatz == ArrayAnsatz(
            math.hermite_renormalized(*displacement_gate_Abc(x=0.1, y=0.1), shape=(4, 6)),
        )
        for w in d_fock.wires.quantum:
            assert w.repr == ReprEnum.FOCK
            assert w.fock_cutoff == d_fock.ansatz.core_shape[w.index]

        d_fock_barg = d_fock.to_bargmann()
        assert d_fock.ansatz._original_abc_data == d.ansatz.triple
        assert d_fock_barg == d
        for w in d_fock_barg.wires.quantum:
            assert w.repr == ReprEnum.BARGMANN

    def test_to_fock_bargmann_poly_exp(self):
        A, b, _ = Abc_triple(3)
        c = settings.rng.random(5) + 0.0j
        polyexp = PolyExpAnsatz(A, b, c)
        fock_cc = CircuitComponent(
            ansatz=polyexp,
            wires=Wires(set(), set(), {0, 1}, set()),
        ).to_fock(shape=(10, 10))
        poly = math.hermite_renormalized(A, b, 1, (10, 10, 5))
        assert fock_cc.ansatz._original_abc_data is None
        assert math.allclose(fock_cc.ansatz.data, math.einsum("ijk,k", poly, c))

        barg_cc = fock_cc.to_bargmann()
        A_expected, b_expected, _ = identity_Abc(2)
        assert math.allclose(barg_cc.ansatz.A, A_expected)
        assert math.allclose(barg_cc.ansatz.b, b_expected)
        assert math.allclose(barg_cc.ansatz.c, fock_cc.ansatz.data)

    def test_add(self):
        cc1 = CircuitComponent.from_bargmann(Abc_triple(1), modes_out_ket=(0,))
        cc2 = CircuitComponent.from_bargmann(Abc_triple(1), modes_out_ket=(0,))

        cc12 = cc1 + cc2

        assert isinstance(cc12, CircuitComponent)
        assert cc12.ansatz == cc1.ansatz + cc2.ansatz
        assert cc12.ansatz._lin_sup is True

    def test_add_built_in(self):
        d1 = Dgate(1, x=0.1, y=0.1, x_trainable=True, x_bounds=(0, 1))
        d2 = Dgate(1, x=0.2, y=0.2, x_trainable=True, x_bounds=(0, 1))

        d12 = d1 + d2

        assert isinstance(d12, Dgate)
        assert isinstance(d12.parameters.x, Variable)
        assert d12.parameters.x.bounds == (0, 1)
        assert math.allclose(d12.parameters.x.value, [0.1, 0.2])
        assert math.allclose(d12.parameters.y.value, [0.1, 0.2])
        assert d12.ansatz._lin_sup is True
        assert d12.ansatz == d1.ansatz + d2.ansatz

    def test_add_error(self):
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.2, y=0.2)
        d3 = Dgate(1, x=0.1, y=0.1, x_trainable=True)
        d4 = Dgate(1, x=0.1, y=0.1, x_trainable=True, x_bounds=(0, 1))
        d_batched = Dgate(1, x=[0.1, 0.2])

        with pytest.raises(ValueError, match="different wires"):
            d1 + d2

        with pytest.raises(ValueError, match="Parameter 'x' is a"):
            d1 + d3

        with pytest.raises(ValueError, match="batched"):
            d1 + d_batched

        with pytest.raises(ValueError, match="Parameter 'x' has bounds"):
            d3 + d4

    def test_sub(self):
        s1 = DisplacedSqueezed(1, x=1.0, y=0.5, r=0.1)
        s2 = DisplacedSqueezed(1, x=0.5, y=0.2, r=0.2)
        s12 = s1 - s2
        assert s12.ansatz == s1.ansatz - s2.ansatz

    def test_mul(self):
        d1 = Dgate(1, x=0.1, y=0.1)

        assert (d1 * 3).ansatz == d1.ansatz * 3
        assert (3 * d1).ansatz == d1.ansatz * 3
        assert isinstance(d1 * 3, Unitary)

    def test_truediv(self):
        d1 = Dgate(1, x=0.1, y=0.1)

        assert (d1 / 3).ansatz == d1.ansatz / 3
        assert isinstance(d1 / 3, Unitary)

    def test_eq(self):
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.1)

        assert d1 == d1._light_copy()
        assert d1 != d2

    def test_contract(self):
        vac012 = Vacuum((0, 1, 2))
        d012 = Dgate(0, x=0.1, y=0.1) >> Dgate(1, x=0.1, y=0.1) >> Dgate(2, x=0.1, y=0.1)
        a0 = Attenuator(0, 0.8)
        a1 = Attenuator(1, 0.8)
        a2 = Attenuator(2, 0.7)

        result = vac012.contract(d012)
        result = result.contract(result.adjoint).contract(a0).contract(a1).contract(a2)

        assert result.wires == Wires(modes_out_bra={0, 1, 2}, modes_out_ket={0, 1, 2})
        assert math.allclose(result.ansatz.A, math.zeros_like(result.ansatz.A))
        assert math.allclose(
            result.ansatz.b,
            [
                [
                    0.08944272 - 0.08944272j,
                    0.08944272 - 0.08944272j,
                    0.083666 - 0.083666j,
                    0.08944272 + 0.08944272j,
                    0.08944272 + 0.08944272j,
                    0.083666 + 0.083666j,
                ],
            ],
        )
        assert math.allclose(result.ansatz.c, [0.95504196])

    def test_contract_one_mode_Dgate(self):
        r"""
        Tests that ``contract`` produces the correct outputs for two Dgate with the formula well-known.
        """
        alpha = 1.5 + 0.7888 * 1j
        beta = -0.1555 + 1j * 2.1

        d1 = Dgate(0, x=alpha.real, y=alpha.imag)
        d2 = Dgate(0, x=beta.real, y=beta.imag)

        result1 = d2.contract(d1)
        correct_c = np.exp(-0.5 * (abs(alpha + beta) ** 2)) * np.exp(
            (alpha * np.conj(beta) - np.conj(alpha) * beta) / 2,
        )

        assert math.allclose(result1.ansatz.c, correct_c)

    def test_contract_is_associative(self):
        d0 = Dgate(0, x=0.1, y=0.1)
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.1)
        a0 = Attenuator(0, transmissivity=0.8)
        a1 = Attenuator(1, transmissivity=0.8)
        a2 = Attenuator(2, transmissivity=0.7)

        result1 = d0.contract(d1).contract(a0).contract(a1).contract(a2).contract(d2)
        result2 = d0.contract(d1.contract(a0)).contract(a1).contract(a2).contract(d2)
        result3 = d0.contract(d1.contract(a0).contract(a1)).contract(a2).contract(d2)
        result4 = d0.contract(d1.contract(a0).contract(a1).contract(a2)).contract(d2)

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_contract_scalar(self):
        d0 = Dgate(0, x=0.1, y=0.1)
        result = d0.contract(0.8)
        assert math.allclose(result.ansatz.A, d0.ansatz.A)
        assert math.allclose(result.ansatz.b, d0.ansatz.b)
        assert math.allclose(result.ansatz.c, 0.8 * d0.ansatz.c)

    def test_contract_diff_representations(self):
        coh0 = Coherent(0, x=0.1, y=0.1)
        coh1 = Coherent(1, x=0.2, y=0.2).to_fock()

        with settings(DEFAULT_REPRESENTATION="Bargmann"):
            result1 = coh0.contract(coh1)
            assert isinstance(result1.ansatz, PolyExpAnsatz)

        with settings(DEFAULT_REPRESENTATION="Fock"):
            result2 = coh0.contract(coh1)
            assert isinstance(result2.ansatz, ArrayAnsatz)

    def test_rshift_all_bargmann(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, x=0.1, y=0.1)
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.1)
        a0 = Attenuator(0, transmissivity=0.8)
        a1 = Attenuator(1, transmissivity=0.8)
        a2 = Attenuator(2, transmissivity=0.7)

        result = vac012 >> d0 >> d1 >> d2 >> a0 >> a1 >> a2

        assert result.wires == Wires(modes_out_bra={0, 1, 2}, modes_out_ket={0, 1, 2})
        assert math.allclose(result.ansatz.A, math.zeros_like(result.ansatz.A))
        assert math.allclose(
            result.ansatz.b,
            [
                [
                    0.08944272 - 0.08944272j,
                    0.08944272 - 0.08944272j,
                    0.083666 - 0.083666j,
                    0.08944272 + 0.08944272j,
                    0.08944272 + 0.08944272j,
                    0.083666 + 0.083666j,
                ],
            ],
        )
        assert math.allclose(result.ansatz.c, [0.95504196])

    def test_rshift_all_fock(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, x=0.1, y=0.1)
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.1)
        a0 = Attenuator(0, transmissivity=0.8)
        a1 = Attenuator(1, transmissivity=0.8)
        a2 = Attenuator(2, transmissivity=0.7)

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
        ).to_fock(N)

        assert r1 == r2

    @pytest.mark.parametrize("shape", [5, 6])
    def test_rshift_bargmann_and_fock(self, shape):
        with settings(AUTOSHAPE_MAX=shape):
            vac12 = Vacuum((1, 2))
            d1 = Dgate(1, x=0.4, y=0.1)
            d2 = Dgate(2, x=0.1, y=0.5)
            a1 = Attenuator(1, transmissivity=0.9)
            n1 = Number(1, n=1).dual
            n2 = Number(2, n=1).dual

            # bargmann >> fock
            r1 = vac12 >> d1 >> d2 >> a1 >> n1 >> n2

            # bargmann >> fock
            r1 = vac12 >> d1 >> d2 >> a1 >> n1 >> n2

            # bargmann >> fock
            r1 = vac12 >> d1 >> d2 >> a1 >> n1 >> n2

            # fock >> bargmann
            r2 = vac12.to_fock(shape) >> d1 >> d2 >> a1 >> n1 >> n2

            # bargmann >> fock >> bargmann
            r3 = vac12 >> d1.to_fock(shape) >> d2 >> a1 >> n1 >> n2

            assert math.allclose(r1, r2)
            assert math.allclose(r1, r3)

    def test_rshift_error(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, x=0.1, y=0.1)
        d0._wires = Wires()

        with pytest.raises(ValueError, match="not clear"):
            vac012 >> d0

    def test_rshift_is_associative(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, x=0.1, y=0.1)
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.1)
        a0 = Attenuator(0, transmissivity=0.8)
        a1 = Attenuator(1, transmissivity=0.8)
        a2 = Attenuator(2, transmissivity=0.7)

        result1 = vac012 >> d0 >> d1 >> a0 >> a1 >> a2 >> d2
        result2 = (vac012 >> d0) >> (d1 >> a0) >> a1 >> (a2 >> d2)
        result3 = vac012 >> (d0 >> (d1 >> a0 >> a1) >> a2 >> d2)
        result4 = vac012 >> (d0 >> (d1 >> (a0 >> (a1 >> (a2 >> d2)))))

        assert result1 == result2
        assert result1 == result3
        assert result1 == result4

    def test_rshift_ketbra_with_ket(self):
        a1 = Attenuator(1, transmissivity=0.8)
        n1 = Number(1, n=1).dual >> Number(2, n=1).dual
        assert a1 >> n1 == a1.contract(n1.contract(n1.adjoint))

    def test_rshift_perm_order(self):
        rng = np.random.default_rng(seed=2334255467567)
        r = [rng.random(), -rng.random()]
        theta = rng.random() * 2 * np.pi
        pnr_cutoff = rng.integers(1, 25)
        out_loss = rng.random()
        pnr_loss = rng.random()

        outcome = rng.integers(0, pnr_cutoff)

        mm_state = (
            SqueezedVacuum(0, r[0])
            >> SqueezedVacuum(1, r[1])
            >> BSgate((0, 1), theta)
            >> Attenuator(0, 1 - out_loss)
            >> Attenuator(1, 1 - pnr_loss)
            >> Number(1, outcome).dual
        )

        mm_state_dm = (
            SqueezedVacuum(0, r[0])
            >> SqueezedVacuum(1, r[1])
            >> BSgate((0, 1), theta)
            >> Attenuator(0, 1 - out_loss)
            >> Attenuator(1, 1 - pnr_loss)
            >> Number(1, outcome).dm().dual
        )
        assert mm_state == mm_state_dm

    def test_rshift_scalar(self):
        d0 = Dgate(0, x=0.1, y=0.1)
        result = 0.8 >> d0
        assert math.allclose(result, 0.8 * d0.ansatz.c)

        result2 = d0 >> 0.8
        assert math.allclose(result2.ansatz.c, 0.8 * d0.ansatz.c)

    def test_repr(self):
        c1 = CircuitComponent(ansatz=None, wires=Wires(modes_out_ket={0, 1, 2}))
        c2 = CircuitComponent(
            ansatz=None,
            wires=Wires(modes_out_ket={0, 1, 2}),
            name="my_component",
        )

        assert repr(c1) == "CircuitComponent(modes=(0, 1, 2), name=CC012)"
        assert repr(c2) == "CircuitComponent(modes=(0, 1, 2), name=my_component)"

    def test_to_fock_shape_lookahead(self):
        r = settings.rng.uniform(-0.5, 0.5, 3)
        interf = Interferometer([0, 1])
        gaussian_part = SqueezedVacuum(0, r[0]) >> SqueezedVacuum(1, r[1]) >> interf
        gauss_auto_shape = gaussian_part.auto_shape()
        fock_explicit_shape = gaussian_part.to_fock((gauss_auto_shape[0], 7)) >> Number(1, 6).dual
        fock_lookahead_shape = gaussian_part >> Number(1, 6).dual
        assert fock_lookahead_shape == fock_explicit_shape

    def test_to_fock_keeps_bargmann(self):
        "tests that to_fock doesn't lose the bargmann representation"
        coh = Coherent(0, x=1.0)
        coh.to_fock(20)
        assert coh.bargmann_triple() == Coherent(0, x=1.0).bargmann_triple()

    def test_fock_component_no_bargmann(self):
        "tests that a fock component doesn't have a bargmann representation by default"
        coh = Coherent(0, x=1.0)
        CC = Ket.from_fock((0,), coh.fock_array(20))
        with pytest.raises(AttributeError):
            CC.bargmann_triple()

    def test_quadrature_ket(self):
        "tests that transforming to quadrature and back gives the same ket"
        ket = SqueezedVacuum(0, 0.4, 0.5) >> Dgate(0, 0.3, 0.2)
        back = Ket.from_quadrature((0,), ket.quadrature_triple())
        assert ket == back

        ket_fock = Number(0, n=1)
        back2 = Ket.from_quadrature((0,), ket_fock.quadrature_triple())
        assert ket_fock.to_bargmann() == back2

    def test_quadrature_channel(self):
        C = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3, 0.2) >> Attenuator(0, 0.9)
        back = Channel.from_quadrature((0,), (0,), C.quadrature_triple())
        assert back == C

    def test_quadrature_dm(self):
        "tests that transforming to quadrature and back gives the same density matrix"
        dm = SqueezedVacuum(0, 0.4, 0.5) >> Dgate(0, 0.3, 0.2) >> Attenuator(0, 0.9)
        back = DM.from_quadrature((0,), dm.quadrature_triple())
        assert dm == back

    def test_quadrature_map(self):
        C = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3, 0.2) >> Attenuator(0, 0.9)
        back = Map.from_quadrature((0,), (0,), C.quadrature_triple())
        assert back == C

    def test_quadrature_operation(self):
        U = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3, 0.2)
        back = Operation.from_quadrature((0,), (0,), U.quadrature_triple())
        assert back == U

    def test_quadrature_unitary(self):
        U = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3, 0.2)
        back = Unitary.from_quadrature((0,), (0,), U.quadrature_triple())
        assert back == U

    @pytest.mark.parametrize("is_fock,widget_cls", [(False, Box), (True, HBox)])
    @patch("mrmustard.lab.circuit_components.display")
    def test_ipython_repr(self, mock_display, is_fock, widget_cls):
        """Test the IPython repr function."""
        dgate = Dgate(1, x=0.1, y=0.1)
        if is_fock:
            dgate = dgate.to_fock()
        dgate._ipython_display_()
        [box] = mock_display.call_args.args
        assert isinstance(box, Box)
        [wires_widget, rep_widget] = box.children
        assert isinstance(wires_widget, HTML)
        assert isinstance(rep_widget, widget_cls)

    @patch("mrmustard.lab.circuit_components.display")
    def test_ipython_repr_invalid_obj(self, mock_display):
        """Test the IPython repr function."""
        dgate = (Dgate(1, x=0.1, y=0.1) >> Dgate(2, x=0.1, y=0.1)).to_fock()
        dgate._ipython_display_()
        [box] = mock_display.call_args.args
        assert isinstance(box, VBox)
        [title_widget, wires_widget] = box.children
        assert isinstance(title_widget, HTML)
        assert isinstance(wires_widget, HTML)

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        dgate = (Dgate(1, x=0.1, y=0.1) >> Dgate(2, x=0.1, y=0.1)).to_fock()
        dgate._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(dgate)

    def test_serialize_default_behaviour(self):
        """Test the default serializer."""
        name = "my_component"
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(0.1, 0.4))
        cc = CircuitComponent(ansatz, Wires(set(), set(), {1, 8}, {1, 8}), name=name)
        kwargs, arrays = cc._serialize()
        assert kwargs == {
            "class": f"{CircuitComponent.__module__}.CircuitComponent",
            "wires": tuple(tuple(w) for w in cc.wires.args),
            "ansatz_cls": f"{PolyExpAnsatz.__module__}.PolyExpAnsatz",
            "name": name,
        }
        assert arrays == {
            "A": ansatz.A,
            "b": ansatz.b,
            "c": ansatz.c,
        }

    def test_serialize_fail_when_no_modes_input(self):
        """Test that the serializer fails if no modes or name+wires are present."""

        class MyComponent(CircuitComponent):
            """A dummy class without a valid modes kwarg."""

            def __init__(self, ansatz, custom_modes):
                super().__init__(
                    ansatz,
                    Wires(*tuple(set(m) for m in [custom_modes] * 4)),
                    name="my_component",
                )

        cc = MyComponent(PolyExpAnsatz(*displacement_gate_Abc(0.1, 0.4)), [0, 1])
        with pytest.raises(
            TypeError,
            match="MyComponent does not seem to have any wires construction method",
        ):
            cc._serialize()

    def test_hermite_renormalized_with_custom_shape(self):
        """Test hermite_renormalized with a custom non-zero shape"""

        S = SqueezedVacuum(0, r=1.0, phi=0, r_trainable=True, phi_trainable=True)

        # made up, means nothing
        def cost():
            ket = S.fock_array(shape=[3])
            return -math.real(ket[2])

        circuit = Circuit([S])

        opt = Optimizer()

        if math.backend_name == "tensorflow":
            assert opt.minimize(cost, by_optimizing=[circuit], max_steps=5) is None
        else:
            with pytest.raises(
                NotImplementedError,
                match="not implemented for backend ``(numpy|jax)``",
            ):
                opt.minimize(cost, by_optimizing=[circuit], max_steps=5)
