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
from mrmustard.parameters import Constant, Variable
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.triples import displacement_gate_Abc, identity_Abc
from mrmustard.physics.utils import random_Abc
from mrmustard.physics.wires import QuantumWire, ReprEnum, Wires


class TestCircuitComponent:
    r"""
    Tests ``CircuitComponent`` objects.
    """

    @pytest.mark.parametrize("x", [0.1, [0.2, 0.3]])
    @pytest.mark.parametrize("y", [0.4, [0.5, 0.6]])
    def test_init(self, x, y):
        name = "my_component"
        x = math.astensor(x, dtype=math.complex128)
        y = math.astensor(y, dtype=math.complex128)
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(x + 1j * y))
        ansatz_factory, _ = AnsatzFactory.from_ansatz(ansatz, ReprEnum.BARGMANN)
        cc = CircuitComponent(
            ansatz_factory=ansatz_factory, wires=Wires(set(), set(), {1, 8}, {1, 8}), name=name
        )
        assert cc.name == name
        assert cc.modes == (1, 8)
        assert cc.wires == Wires(modes_out_ket={1, 8}, modes_in_ket={1, 8})
        assert cc.ansatz == ansatz
        assert cc.manual_shape == (None,) * 4

    def test_missing_name(self):
        ansatz_factory, _ = AnsatzFactory.from_ansatz(
            PolyExpAnsatz(*displacement_gate_Abc(0.1 + 0.2j)), ReprEnum.BARGMANN
        )
        cc = CircuitComponent(
            ansatz_factory=ansatz_factory,
            wires=Wires(set(), set(), {1, 8}, {1, 8}),
        )
        cc._name = None
        assert cc.name == "CC18"

    def test_from_bargmann(self):
        cc = CircuitComponent.from_bargmann(displacement_gate_Abc(0.1 + 0.2j), {}, {}, {0}, {0})
        assert cc.ansatz == PolyExpAnsatz(*displacement_gate_Abc(0.1 + 0.2j))

    def test_from_attributes(self):
        cc = Dgate(1, alpha=0.1 + 0.2j)

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
        c = Dgate(0, alpha=0.1 + 0.2j) >> Sgate(0, r=1.0, phi=0.1)
        ansatz_factory, _ = AnsatzFactory.from_ansatz(c.ansatz, ReprEnum.BARGMANN)
        cc = CircuitComponent(ansatz_factory=ansatz_factory, wires=c.wires, name=c.name)
        ccc = CircuitComponent.from_quadrature((), (), (0,), (0,), cc.quadrature_triple())
        assert cc == ccc

    def test_adjoint(self):
        d1 = Dgate(1, alpha=0.1 + 0.2j)
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
        d1 = Dgate(1, alpha=0.1 + 0.2j)
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
        ansatz = PolyExpAnsatz(*displacement_gate_Abc(0.1 + 0.1j))
        wires = Wires(set(), set(), {1}, {1})
        ansatz_factory, _ = AnsatzFactory.from_ansatz(ansatz, ReprEnum.BARGMANN)
        d1 = CircuitComponent(ansatz_factory=ansatz_factory, wires=wires)
        d1_cp = d1._light_copy()

        assert d1_cp.parameters is d1.parameters
        assert d1_cp.ansatz is d1.ansatz
        assert d1_cp.wires is not d1.wires

    def test_on(self):
        assert Vacuum([1, 2]).on([3, 4]).modes == (3, 4)
        assert Number(3, n=4).on(9).modes == (9,)

        r_var = Variable(0, "r", dtype=math.float64)
        d8 = DisplacedSqueezed(8, alpha=1 + 3j, r=r_var)
        d6 = d8.on(6)
        assert isinstance(d6.parameters.alpha, Constant)
        assert math.allclose(d8.parameters.alpha.value, d6.parameters.alpha.value)
        assert isinstance(d6.parameters.r, Variable)
        assert math.allclose(d8.parameters.r.value, d6.parameters.r.value)
        assert bool(d6.parameters) is True
        assert d6.ansatz is d8.ansatz

        # ensure that representation and fock shape are preserved
        d8_fock = d8.to_fock()
        d6_fock_on = d8_fock.on(6)
        for w8, w6 in zip(d8_fock.wires, d6_fock_on.wires):
            assert w8.repr == w6.repr
            assert w8.fock_shape == w6.fock_shape

        # ensure that on matches to_fock
        d6_fock_expected = d6.to_fock()
        for w6, w6_expected in zip(d6_fock_on.wires, d6_fock_expected.wires):
            assert w6.repr == w6_expected.repr
            assert w6.fock_shape == w6_expected.fock_shape

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
        d = Dgate(1, alpha=0.1 + 0.1j)
        d_barg = d.to_bargmann()
        assert d is d_barg

        d_fock = d.to_fock(shape=(4, 6))
        assert d_fock.ansatz == ArrayAnsatz(
            math.hermite_renormalized(*displacement_gate_Abc(0.1 + 0.1j), shape=(4, 6)),
        )
        for w in d_fock.wires.quantum:
            assert w.repr == ReprEnum.FOCK
            assert w.fock_shape == d_fock.ansatz.core_shape[w.index]

        d_fock_barg = d_fock.to_bargmann()
        assert d_fock_barg == d
        for w in d_fock_barg.wires.quantum:
            assert w.repr == ReprEnum.BARGMANN

    def test_to_fock_bargmann_poly_exp(self):
        A, b, _ = random_Abc(3)
        c = settings.get_rng().random(5) + 0.0j
        polyexp = PolyExpAnsatz(A, b, c)
        ansatz_factory, _ = AnsatzFactory.from_ansatz(polyexp, ReprEnum.BARGMANN)
        fock_cc = CircuitComponent(
            ansatz_factory=ansatz_factory,
            wires=Wires(set(), set(), {0, 1}, set()),
        ).to_fock(shape=(10, 10))
        poly = math.hermite_renormalized(A, b, 1, (10, 10, 5))
        assert math.allclose(fock_cc.ansatz.data, math.einsum("ijk,k", poly, c))

        barg_cc = fock_cc.to_bargmann()
        A_expected, b_expected, _ = identity_Abc(2)
        assert math.allclose(barg_cc.ansatz.A, A_expected)
        assert math.allclose(barg_cc.ansatz.b, b_expected)
        assert math.allclose(barg_cc.ansatz.c, fock_cc.ansatz.data)

    def test_add(self):
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(1, alpha=0.2 + 0.2j)

        d12 = d1 + d2

        assert d12.ansatz._lin_sup is True
        assert d12.ansatz == d1.ansatz + d2.ansatz

    def test_add_error(self):
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.2 + 0.2j)
        d_batched = Dgate(1, alpha=[0.1, 0.2])

        with pytest.raises(ValueError, match="different wires"):
            d1 + d2

        with pytest.raises(ValueError, match="Cannot add PolyExpAnsatz"):
            d1 + d_batched

    def test_sub(self):
        s1 = DisplacedSqueezed(1, alpha=1.0 + 0.5j, r=0.1)
        s2 = DisplacedSqueezed(1, alpha=0.5 + 0.2j, r=0.2)
        s12 = s1 - s2
        assert s12.ansatz == s1.ansatz - s2.ansatz

    def test_mul(self):
        d1 = Dgate(1, alpha=0.1 + 0.1j)

        assert (d1 * 3).ansatz == d1.ansatz * 3
        assert (3 * d1).ansatz == d1.ansatz * 3
        assert isinstance(d1 * 3, Unitary)

    def test_truediv(self):
        d1 = Dgate(1, alpha=0.1 + 0.1j)

        assert (d1 / 3).ansatz == d1.ansatz / 3
        assert isinstance(d1 / 3, Unitary)

    def test_eq(self):
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.1 + 0.1j)

        assert d1 == d1._light_copy()
        assert d1 != d2

    def test_contract(self):
        vac012 = Vacuum((0, 1, 2))
        d012 = (
            Dgate(0, alpha=0.1 + 0.1j) >> Dgate(1, alpha=0.1 + 0.1j) >> Dgate(2, alpha=0.1 + 0.1j)
        )
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

        d1 = Dgate(0, alpha)
        d2 = Dgate(0, beta)

        result1 = d2.contract(d1)
        correct_c = np.exp(-0.5 * (abs(alpha + beta) ** 2)) * np.exp(
            (alpha * np.conj(beta) - np.conj(alpha) * beta) / 2,
        )

        assert math.allclose(result1.ansatz.c, correct_c)

    def test_contract_scalar(self):
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        result = d0.contract(0.8)
        assert math.allclose(result.ansatz.A, d0.ansatz.A)
        assert math.allclose(result.ansatz.b, d0.ansatz.b)
        assert math.allclose(result.ansatz.c, 0.8 * d0.ansatz.c)

    def test_matmul_is_associative(self):
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.1 + 0.1j)
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

    def test_rmatmul(self):
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        result = 0.8 @ d0
        assert math.allclose(result.ansatz.A, d0.ansatz.A)
        assert math.allclose(result.ansatz.b, d0.ansatz.b)
        assert math.allclose(result.ansatz.c, 0.8 * d0.ansatz.c)

    def test_contract_diff_representations(self):
        coh0 = Coherent(0, alpha=0.1 + 0.1j)
        coh1 = Coherent(1, alpha=0.2 + 0.2j).to_fock()

        with settings(DEFAULT_REPRESENTATION="Bargmann"):
            result1 = coh0.contract(coh1)
            assert isinstance(result1.ansatz, PolyExpAnsatz)

        with settings(DEFAULT_REPRESENTATION="Fock"):
            result2 = coh0.contract(coh1)
            assert isinstance(result2.ansatz, ArrayAnsatz)

        with pytest.raises(TypeError), settings(DEFAULT_REPRESENTATION=None):
            coh0.contract(coh1)

    def test_to_fock_shape_error(self):
        state = Coherent(0, alpha=0.1 + 0.1j)
        with pytest.raises(ValueError, match="non-zero"):
            state.to_fock(shape=(0, 1))
        with pytest.raises(ValueError, match="Fock shape of"):
            state.to_fock(shape=(1, 1, 1, 1))

    def test_rshift_all_bargmann(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.1 + 0.1j)
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
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.1 + 0.1j)
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
            d1 = Dgate(1, alpha=0.4 + 0.1j)
            d2 = Dgate(2, alpha=0.1 + 0.5j)
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
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        d0._wires = Wires()

        with pytest.raises(ValueError, match="not clear"):
            vac012 >> d0

    def test_rshift_is_associative(self):
        vac012 = Vacuum((0, 1, 2))
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        d1 = Dgate(1, alpha=0.1 + 0.1j)
        d2 = Dgate(2, alpha=0.1 + 0.1j)
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
        d0 = Dgate(0, alpha=0.1 + 0.1j)
        result = 0.8 >> d0
        assert math.allclose(result, 0.8 * d0.ansatz.c)

        result2 = d0 >> 0.8
        assert math.allclose(result2.ansatz.c, 0.8 * d0.ansatz.c)

    def test_repr(self):
        c1 = CircuitComponent(ansatz_factory=None, wires=Wires(modes_out_ket={0, 1, 2}))
        c2 = CircuitComponent(
            ansatz_factory=None,
            wires=Wires(modes_out_ket={0, 1, 2}),
            name="my_component",
        )

        assert repr(c1) == "CircuitComponent(modes=(0, 1, 2), name=CC012)"
        assert repr(c2) == "CircuitComponent(modes=(0, 1, 2), name=my_component)"

    def test_to_fock_shape_lookahead(self):
        r = settings.get_rng().uniform(-0.5, 0.5, 3)
        interf = Interferometer.random(modes=(0, 1))
        gaussian_part = SqueezedVacuum(0, r[0]) >> SqueezedVacuum(1, r[1]) >> interf
        gauss_auto_shape = gaussian_part.auto_shape()
        fock_explicit_shape = gaussian_part.to_fock((gauss_auto_shape[0], 7)) >> Number(1, 6).dual
        fock_lookahead_shape = gaussian_part >> Number(1, 6).dual
        assert fock_lookahead_shape == fock_explicit_shape

    def test_to_fock_keeps_bargmann(self):
        "tests that to_fock doesn't lose the bargmann representation"
        coh = Coherent(0, alpha=1.0)
        coh.to_fock(20)
        assert coh.bargmann_triple() == Coherent(0, alpha=1.0).bargmann_triple()

    def test_to_standard_order(self):
        atten = Attenuator(0, transmissivity=0.5)
        atten_dual = atten.dual
        atten_dual_so = atten_dual.to_standard_order()

        # before standard order
        assert atten_dual.wires.standard_order == [
            QuantumWire(mode=0, is_out=True, is_ket=False, index=1),
            QuantumWire(mode=0, is_out=False, is_ket=False, index=0),
            QuantumWire(mode=0, is_out=True, is_ket=True, index=3),
            QuantumWire(mode=0, is_out=False, is_ket=True, index=2),
        ]

        # after standard order
        assert atten_dual_so.wires.standard_order == [
            QuantumWire(mode=0, is_out=True, is_ket=False, index=0),
            QuantumWire(mode=0, is_out=False, is_ket=False, index=1),
            QuantumWire(mode=0, is_out=True, is_ket=True, index=2),
            QuantumWire(mode=0, is_out=False, is_ket=True, index=3),
        ]

    def test_fock_component_no_bargmann(self):
        "tests that a fock component doesn't have a bargmann representation by default"
        coh = Coherent(0, alpha=1.0)
        CC = Ket.from_fock((0,), coh.fock_array(20))
        with pytest.raises(AttributeError, match="No Bargmann data for this component."):
            CC.bargmann_triple()

    def test_quadrature_ket(self):
        "tests that transforming to quadrature and back gives the same ket"
        ket = SqueezedVacuum(0, 0.4, 0.5) >> Dgate(0, 0.3 + 0.2j)
        back = Ket.from_quadrature((0,), ket.quadrature_triple())
        assert ket == back

        ket_fock = Number(0, n=1)
        back2 = Ket.from_quadrature((0,), ket_fock.quadrature_triple())
        assert ket_fock.to_bargmann() == back2

    def test_quadrature_channel(self):
        C = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3 + 0.2j) >> Attenuator(0, 0.9)
        back = Channel.from_quadrature((0,), (0,), C.quadrature_triple())
        assert back == C

    def test_quadrature_dm(self):
        "tests that transforming to quadrature and back gives the same density matrix"
        dm = SqueezedVacuum(0, 0.4, 0.5) >> Dgate(0, 0.3 + 0.2j) >> Attenuator(0, 0.9)
        back = DM.from_quadrature((0,), dm.quadrature_triple())
        assert dm == back

    def test_quadrature_map(self):
        C = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3 + 0.2j) >> Attenuator(0, 0.9)
        back = Map.from_quadrature((0,), (0,), C.quadrature_triple())
        assert back == C

    def test_quadrature_operation(self):
        U = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3 + 0.2j)
        back = Operation.from_quadrature((0,), (0,), U.quadrature_triple())
        assert back == U

    def test_quadrature_unitary(self):
        U = Sgate(0, 0.5, 0.4) >> Dgate(0, 0.3 + 0.2j)
        back = Unitary.from_quadrature((0,), (0,), U.quadrature_triple())
        assert back == U

    @pytest.mark.parametrize("is_fock,widget_cls", [(False, Box), (True, HBox)])
    @patch("mrmustard.lab.circuit_components.display")
    def test_ipython_repr(self, mock_display, is_fock, widget_cls):
        """Test the IPython repr function."""
        dgate = Dgate(1, alpha=0.1 + 0.1j)
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
        dgate = (Dgate(1, alpha=0.1 + 0.1j) >> Dgate(2, alpha=0.1 + 0.1j)).to_fock()
        dgate._ipython_display_()
        [box] = mock_display.call_args.args
        assert isinstance(box, VBox)
        [title_widget, wires_widget] = box.children
        assert isinstance(title_widget, HTML)
        assert isinstance(wires_widget, HTML)

    @patch("mrmustard.widgets.IN_INTERACTIVE_SHELL", True)
    def test_ipython_repr_interactive(self, capsys):
        """Test the IPython repr function."""
        dgate = (Dgate(1, alpha=0.1 + 0.1j) >> Dgate(2, alpha=0.1 + 0.1j)).to_fock()
        dgate._ipython_display_()
        captured = capsys.readouterr()
        assert captured.out.rstrip() == repr(dgate)

    def test_circuit_component_batch_getitem_array_ansatz(self):
        array = math.arange(2 * 3 * 4).reshape((2, 3, 4))
        ansatz_factory, _ = AnsatzFactory.from_ansatz(
            ArrayAnsatz(array, batch_dims=1), ReprEnum.FOCK
        )
        wires = Wires({0}, set(), set(), set())
        for w in wires.quantum:
            w.repr = ReprEnum.FOCK
            w.fock_shape = array.shape[w.index]
        cc = CircuitComponent(ansatz_factory=ansatz_factory, wires=wires)
        sub = cc[1]
        assert isinstance(sub, CircuitComponent)
        assert isinstance(sub.ansatz, ArrayAnsatz)
        assert sub.ansatz.batch_dims == 0

    def test_circuit_component_batch_getitem_polyexp_ansatz(self):
        A, b, c = random_Abc(3, (2,))
        ansatz_factory, _ = AnsatzFactory.from_ansatz(PolyExpAnsatz(A, b, c), ReprEnum.BARGMANN)
        cc = CircuitComponent(ansatz_factory=ansatz_factory, wires=Wires({0}, set(), set(), set()))
        sub = cc[1]
        assert isinstance(sub, CircuitComponent)
        assert isinstance(sub.ansatz, PolyExpAnsatz)
        assert sub.ansatz.batch_dims == 0

    def test_concat_basic_with_bargmann(self):
        """Test basic concatenation of circuit components with Bargmann ansatz."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)

        # Add batch dimensions
        coh1_batched = coh1[None]
        coh2_batched = coh2[None]

        concatenated = coh1_batched.concat(coh2_batched, axis=0)

        assert concatenated.ansatz.batch_shape == (2,)
        assert concatenated.wires == coh1.wires
        assert isinstance(concatenated, Ket)
        assert math.allclose(concatenated.ansatz.A[0], coh1.ansatz.A)
        assert math.allclose(concatenated.ansatz.A[1], coh2.ansatz.A)

    def test_concat_basic_with_fock(self):
        """Test basic concatenation of circuit components with Fock ansatz."""
        num1 = Number(mode=0, n=1).to_fock(5)
        num2 = Number(mode=0, n=2).to_fock(5)

        # Add batch dimensions
        num1_batched = num1[None]
        num2_batched = num2[None]

        concatenated = num1_batched.concat(num2_batched, axis=0)

        assert concatenated.ansatz.batch_shape == (2,)
        assert concatenated.wires == num1.wires
        assert isinstance(concatenated, Ket)
        assert isinstance(concatenated.ansatz, ArrayAnsatz)

    def test_concat_preserves_class_type(self):
        """Test that concat preserves the circuit component class type."""
        dgate1 = Dgate(mode=0, alpha=0.1)
        dgate2 = Dgate(mode=0, alpha=0.2)

        # Add batch dimensions
        dgate1_batched = dgate1[None]
        dgate2_batched = dgate2[None]

        concatenated = dgate1_batched.concat(dgate2_batched, axis=0)

        assert isinstance(concatenated, Unitary)
        assert concatenated.ansatz.batch_shape == (2,)

    def test_concat_multidimensional_batch(self):
        """Test concatenation with multi-dimensional batch shapes."""
        A1, b1, c1 = random_Abc(2, (3, 2))
        A2, b2, c2 = random_Abc(2, (3, 5))
        wires = Wires(set(), set(), {0}, {0})
        ansatz_factory1, _ = AnsatzFactory.from_ansatz(PolyExpAnsatz(A1, b1, c1), ReprEnum.BARGMANN)
        ansatz_factory2, _ = AnsatzFactory.from_ansatz(PolyExpAnsatz(A2, b2, c2), ReprEnum.BARGMANN)
        cc1 = CircuitComponent(ansatz_factory=ansatz_factory1, wires=wires)
        cc2 = CircuitComponent(ansatz_factory=ansatz_factory2, wires=wires)

        concatenated = cc1.concat(cc2, axis=1)

        assert concatenated.ansatz.batch_shape == (3, 7)
        assert concatenated.wires == cc1.wires

    def test_concat_wires_mismatch_error(self):
        """Test that concat fails when wires don't match."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=1, alpha=2.0)  # Different mode

        coh1_batched = coh1[None]
        coh2_batched = coh2[None]

        with pytest.raises(ValueError, match="different wires"):
            coh1_batched.concat(coh2_batched, axis=0)

    def test_concat_ansatz_type_mismatch_error(self):
        """Test that concat fails when ansatz types don't match."""
        coh = Coherent(mode=0, alpha=1.0)
        coh_fock = coh.to_fock(5)

        coh_batched = coh[None]
        coh_fock_batched = coh_fock[None]

        with pytest.raises(ValueError, match="different ansatz types"):
            coh_batched.concat(coh_fock_batched, axis=0)

    def test_stack_basic_with_bargmann(self):
        """Test basic stacking of circuit components with Bargmann ansatz."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)

        stacked = coh1.stack(coh2, axis=0)

        assert stacked.ansatz.batch_shape == (2,)
        assert stacked.wires == coh1.wires
        assert isinstance(stacked, Ket)
        assert math.allclose(stacked.ansatz.A[0], coh1.ansatz.A)
        assert math.allclose(stacked.ansatz.A[1], coh2.ansatz.A)

    def test_stack_basic_with_fock(self):
        """Test basic stacking of circuit components with Fock ansatz."""
        num1 = Number(mode=0, n=1).to_fock(5)
        num2 = Number(mode=0, n=2).to_fock(5)

        stacked = num1.stack(num2, axis=0)

        assert stacked.ansatz.batch_shape == (2,)
        assert stacked.wires == num1.wires
        assert isinstance(stacked, Ket)
        assert isinstance(stacked.ansatz, ArrayAnsatz)

    def test_stack_preserves_class_type(self):
        """Test that stack preserves the circuit component class type."""
        dgate1 = Dgate(mode=0, alpha=0.1)
        dgate2 = Dgate(mode=0, alpha=0.2)

        stacked = dgate1.stack(dgate2, axis=0)

        assert isinstance(stacked, Unitary)
        assert stacked.ansatz.batch_shape == (2,)

    def test_stack_multiple_components(self):
        """Test stacking multiple components sequentially."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)
        coh3 = Coherent(mode=0, alpha=3.0)

        # Stack first two
        stacked12 = coh1.stack(coh2, axis=0)
        # Add batch dimension and concatenate with third
        coh3_batched = coh3[None]
        stacked123 = stacked12.concat(coh3_batched, axis=0)

        assert stacked123.ansatz.batch_shape == (3,)
        assert isinstance(stacked123, Ket)

    def test_stack_wires_mismatch_error(self):
        """Test that stack fails when wires don't match."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=1, alpha=2.0)  # Different mode

        with pytest.raises(ValueError, match="different wires"):
            coh1.stack(coh2, axis=0)

    def test_stack_ansatz_type_mismatch_error(self):
        """Test that stack fails when ansatz types don't match."""
        coh = Coherent(mode=0, alpha=1.0)
        coh_fock = coh.to_fock(5)

        with pytest.raises(ValueError, match="different ansatz types"):
            coh.stack(coh_fock, axis=0)

    def test_concat_and_stack_integration(self):
        """Test integration of concat and stack operations."""
        # Create a batch of 3 coherent states using stack
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)
        coh3 = Coherent(mode=0, alpha=3.0)

        batch_12 = coh1.stack(coh2, axis=0)
        batch_123 = batch_12.concat(coh3[None], axis=0)

        assert batch_123.ansatz.batch_shape == (3,)

        # Test that we can index and get back individual components
        coh1_recovered = batch_123[0]
        assert coh1_recovered.ansatz.batch_dims == 0
        assert math.allclose(coh1_recovered.ansatz.A, coh1.ansatz.A, atol=1e-10)

    def test_stack_batched_components(self):
        """Test stacking batched components."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)
        coh3 = Coherent(mode=0, alpha=3.0)

        # First create two batched components with the same batch shape
        batch_12 = coh1.stack(coh2, axis=0)  # batch_shape (2,)
        batch_13 = coh1.stack(coh3, axis=0)  # batch_shape (2,)

        # Stack the two batched components
        stacked = batch_12.stack(batch_13, axis=0)
        assert stacked.ansatz.batch_shape == (2, 2)  # 2 components, each with batch_shape (2,)
        assert isinstance(stacked, Ket)

        # Test stacking with axis=0 (the only valid axis for unbatched components)
        batch_12_axis0 = coh1.stack(coh2, axis=0)
        assert batch_12_axis0.ansatz.batch_shape == (2,)  # New dimension at axis=0

        # Verify we can recover the original components
        coh1_recovered = batch_12[0]
        coh2_recovered = batch_12[1]

        # Check that recovered components are unbatched
        assert coh1_recovered.ansatz.batch_dims == 0
        assert coh2_recovered.ansatz.batch_dims == 0

        # Check that the representations match the originals
        assert math.allclose(coh1_recovered.ansatz.A, coh1.ansatz.A, atol=1e-10)
        assert math.allclose(coh2_recovered.ansatz.A, coh2.ansatz.A, atol=1e-10)

    def test_stack_with_negative_axis(self):
        """Test stacking with a negative axis."""
        coh1 = Coherent(mode=0, alpha=1.0)
        coh2 = Coherent(mode=0, alpha=2.0)
        coh3 = Coherent(mode=0, alpha=3.0)

        # Create two batched components with the same batch shape
        batch_12 = coh1.stack(coh2, axis=0)  # batch_shape (2,)
        batch_13 = coh1.stack(coh3, axis=0)  # batch_shape (2,)

        # Stack them with negative axis
        stacked = batch_12.stack(batch_13, axis=-1)
        assert stacked.ansatz.batch_shape == (2, 2)  # New axis at the end
        assert isinstance(stacked, Ket)
