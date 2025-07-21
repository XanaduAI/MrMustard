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

"""Tests for the ket."""

import numpy as np
import pytest
from ipywidgets import HTML, Box, HBox, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math, settings
from mrmustard.lab import (
    DM,
    Attenuator,
    BSgate,
    CircuitComponent,
    Coherent,
    Dgate,
    Ggate,
    Identity,
    Ket,
    Number,
    QuadratureEigenstate,
    SqueezedVacuum,
    Vacuum,
)

# Representation class has been removed - functionality moved to CircuitComponent
from mrmustard.physics.triples import coherent_state_Abc
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.physics.wires import Wires
from mrmustard.widgets import state as state_widget

from ...random import Abc_triple


def coherent_state_quad(q, x, y, phi=0):
    """From https://en.wikipedia.org/wiki/Coherent_state#The_wavefunction_of_a_coherent_state"""
    scale = np.sqrt(2 * settings.HBAR)
    alpha = (x + 1j * y) * np.exp(-1j * phi)
    phase = np.sin(2 * phi + 2 * np.arctan2(x, y)) * (x**2 + y**2) / 2
    return (
        math.exp(-1j * phase)  # This global phase allows Coherent >> BtoQ to be equal
        * math.exp(1j * q * alpha.imag * 2 / scale)
        * math.exp(-((q - scale * alpha.real) ** 2) / (scale**2))
        / (np.pi * settings.HBAR) ** 0.25
    )


class TestKet:
    r"""
    Tests for the ``Ket`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_ket"])
    @pytest.mark.parametrize("modes", [(0,), (0, 1), (2, 3, 19)])
    def test_init(self, name, modes):
        state = Ket.from_ansatz(modes, None, name)

        assert name if name else state.name in ("Ket0", "Ket01", "Ket2319")
        assert state.modes == modes
        assert state.wires == Wires(modes_out_ket=set(modes))

    def test_manual_shape(self):
        ket = Coherent(0, x=1)
        assert ket.manual_shape == (None,)
        ket.manual_shape = (19,)
        assert ket.manual_shape == (19,)

    def test_auto_shape(self):
        ket = Coherent(0, x=1)
        assert ket.auto_shape() == (8,)

        ket.manual_shape = (19,)
        assert ket.auto_shape() == (19,)
        assert ket.auto_shape(respect_manual_shape=False) == (8,)

        ket = Coherent(0, x=1) >> Number(1, 10).dual
        assert ket.auto_shape() == (8, 11)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_to_from_bargmann(self, modes, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)

        state_in = Coherent(modes, x, y)
        A, b, c = state_in.bargmann_triple()
        A_expected, b_expected, c_expected = coherent_state_Abc(x, y)

        assert math.allclose(A, A_expected)
        assert math.allclose(b, b_expected)
        assert math.allclose(c, c_expected)

        state_out = Ket.from_bargmann((modes,), (A, b, c), "my_ket")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = Coherent(0, 1) >> Coherent(1, 2)
        with pytest.raises(ValueError):
            Ket.from_bargmann((0,), state01.bargmann_triple(), "my_ket")

    def test_bargmann_triple_error(self):
        with pytest.raises(AttributeError):
            Number(0, n=10).bargmann_triple()

    @pytest.mark.parametrize("coeff", [0.5, 0.3])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_normalize(self, coeff, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x, y)
        state = coeff * state
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)
        fock_state = state.to_fock(5)
        normalized = fock_state.normalize()
        assert math.allclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("coeff", [0.5, 0.3])
    def test_normalize_lin_sup(self, coeff):
        state = Coherent(0, 1, 1) + Coherent(0, -1, -1)
        state = coeff * state
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)
        fock_state = state.to_fock(5)
        normalized = fock_state.normalize()
        assert math.allclose(normalized.probability, 1.0)

        # batch
        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        lin_sup_state_batch = Ket.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        normalized_batch = lin_sup_state_batch.normalize()
        assert math.allclose(normalized_batch.probability, 1.0)
        fock_state = state.to_fock(5)
        normalized = fock_state.normalize()
        assert math.allclose(normalized.probability, 1.0)

    def test_normalize_poly_dim(self):
        # https://github.com/XanaduAI/MrMustard/issues/481
        with settings(AUTOSHAPE_PROBABILITY=0.99999999):
            state = (
                SqueezedVacuum(mode=0, r=0.75)
                >> SqueezedVacuum(mode=1, r=-0.75)
                >> BSgate(modes=(0, 1), theta=0.9)
            ) >> Number(mode=0, n=20).dual

        state = state.to_bargmann().normalize()

        # # Breed 1st round
        state2 = (
            (state.on(0) >> state.on(1))
            >> BSgate(modes=(0, 1), theta=np.pi / 4)
            >> QuadratureEigenstate(mode=1, phi=np.pi / 2).dual
        )

        # # Breed 2nd round
        state3 = (
            (state2.on(0) >> state2.on(1))
            >> BSgate(modes=(0, 1), theta=np.pi / 4)
            >> QuadratureEigenstate(mode=1, phi=np.pi / 2).dual
        )
        state3 = state3.normalize()
        assert math.allclose(state3.probability, 1.0)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_to_from_fock(self, modes, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state_in = Coherent(modes, x=x, y=y)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock_array(5)

        assert math.allclose(array_in, state_in_fock.ansatz.array)

        state_out = Ket.from_fock(
            (modes,),
            array_in,
            "my_ket",
            batch_dims=state_in_fock.ansatz.batch_dims,
        )
        assert state_in_fock == state_out

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_phase_space(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x=x, y=y)
        cov, means, coeff = state.phase_space(s=0)
        assert cov.shape[:-2] == state.ansatz.batch_shape
        assert means.shape[:-1] == state.ansatz.batch_shape
        assert coeff.shape == state.ansatz.batch_shape
        assert math.allclose(coeff, 1.0)
        assert math.allclose(cov, math.eye(2) * settings.HBAR / 2)
        assert math.allclose(means, math.astensor([1.0, 2.0]) * math.sqrt(2 * settings.HBAR))

    @pytest.mark.parametrize("modes", [(0,), (0, 1), (2, 3, 19)])
    def test_from_phase_space(self, modes):
        rnd = Ket.random(modes)
        cov, means, coeff = rnd.phase_space(s=0)
        rnd2 = Ket.from_phase_space(modes, (cov, means, coeff))
        assert rnd == rnd2

        rnd = DM.random(modes)
        cov, means, coeff = rnd.phase_space(s=0)
        rnd2 = DM.from_phase_space(modes, (cov, means, coeff))
        assert rnd == rnd2

    def test_to_from_phase_space(self):
        modes = (0,)
        state = Coherent(0, x=1, y=2)
        cov, means, coeff = state.phase_space(s=0)
        state2 = Ket.from_phase_space(modes, (cov, means, coeff))
        assert state == state2

    @pytest.mark.parametrize("modes", [(0,), (0, 1), (2, 3, 19)])
    @pytest.mark.parametrize("batch_shape", [(1,), (2, 3)])
    def test_to_from_quadrature(self, modes, batch_shape):
        A, b, c = Abc_triple(len(modes), batch_shape)
        state0 = Ket.from_bargmann(modes, (A, b, c))
        Atest, btest, ctest = state0.quadrature_triple()
        state1 = Ket.from_quadrature(modes, (Atest, btest, ctest))
        Atest2, btest2, ctest2 = state1.bargmann_triple()
        assert math.allclose(Atest2, A)
        assert math.allclose(btest2, b)
        assert math.allclose(ctest2, c)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_L2_norm(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x=x, y=y)
        assert math.allclose(state.L2_norm, 1)

    def test_L2_norm_lin_sup(self):
        state = Coherent(mode=0, x=0.1) + Coherent(mode=0, x=0.2)
        L2_norm = state.L2_norm
        assert L2_norm.shape == ()
        assert math.allclose(L2_norm, 3.99002496)

        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        state_batch = Ket.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        L2_norm = state_batch.L2_norm
        assert L2_norm.shape == (3,)
        assert math.allclose(L2_norm, 3.99002496)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_probability(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x=x, y=y) / 3
        probability = state.probability
        assert probability.shape == state.ansatz.batch_shape
        assert math.allclose(probability, 1 / 9)
        assert math.allclose(state.to_fock(20).probability, 1 / 9)

    def test_probability_lin_sup(self):
        state = Coherent(0, x=1) / 2**0.5 + Coherent(0, x=-1) / 2**0.5
        probability = state.probability
        assert probability.shape == ()
        assert math.allclose(probability, 1.13533528)
        assert math.allclose(state.to_fock(20).probability, 1.13533528)

        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        state_batch = Ket.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)

        probability_batch = state_batch.probability
        assert probability_batch.shape == (3,)
        assert math.allclose(probability_batch, 1.13533528)
        assert math.allclose(state_batch.to_fock(20).probability, 1.13533528)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_purity(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x=x, y=y)
        purity = state.purity
        assert purity.shape == state.ansatz.batch_shape
        assert math.allclose(purity, 1)
        assert state.is_pure

        assert math.allclose(Ket.from_ansatz((0,), None).purity, 1)

    def test_purity_lin_sup(self):
        state = Coherent(0, x=1) + Coherent(0, x=-1)
        purity = state.purity
        assert purity.shape == ()
        assert math.allclose(purity, 1)
        assert state.is_pure

        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        state_batch = Ket.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        purity_batch = state_batch.purity
        assert purity_batch.shape == (3,)
        assert math.allclose(purity_batch, 1)
        assert state_batch.is_pure

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_dm(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        ket = Coherent(0, x=x, y=y)
        dm = ket.dm()

        assert dm.ansatz.batch_shape == ket.ansatz.batch_shape
        assert dm.name == ket.name
        assert dm.ansatz == (ket.contract(ket.adjoint, "zip")).ansatz
        assert dm.wires == (ket.contract(ket.adjoint, "zip")).wires

    def test_dm_lin_sup(self):
        state = Coherent(0, x=1) + Coherent(0, x=-1)
        dm = state.dm()
        assert dm.ansatz.batch_shape == (4,)
        assert dm.name == state.name
        assert dm.wires == (state.contract(state.adjoint, "zip")).wires

    @pytest.mark.parametrize("phi", [0, 0.3, np.pi / 4, np.pi / 2])
    def test_quadrature_single_mode_ket(self, phi):
        x, y = 1, 2
        state = Coherent(mode=0, x=x, y=y)
        q = np.linspace(-10, 10, 100)
        psi_phi = coherent_state_quad(q, x, y, phi)
        assert math.allclose(state.quadrature(q, phi=phi), psi_phi)
        assert math.allclose(state.quadrature_distribution(q, phi=phi), abs(psi_phi) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(q, phi=phi), psi_phi)
        assert math.allclose(
            state.to_fock(40).quadrature_distribution(q, phi=phi),
            abs(psi_phi) ** 2,
        )

    def test_quadrature_multimode_ket(self):
        x, y = 1, 2
        state = Coherent(0, x=x, y=y) >> Coherent(1, x=x, y=y)
        q = np.linspace(-10, 10, 100)
        psi_q = math.kron(coherent_state_quad(q, x, y), coherent_state_quad(q, x, y))
        assert math.allclose(state.quadrature(q, q), psi_q)
        assert math.allclose(state.quadrature_distribution(q), abs(psi_q) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(q, q), psi_q)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), abs(psi_q) ** 2)

    def test_quadrature_multivariable_ket(self):
        x, y = 1, 2
        state = Coherent(0, x=x, y=y) >> Coherent(1, x=x, y=y)
        q1 = np.linspace(-10, 10, 100)
        q2 = np.linspace(-10, 10, 100)
        psi_q = math.outer(coherent_state_quad(q1, x, y), coherent_state_quad(q2, x, y))
        assert math.allclose(
            state.quadrature_distribution(q1, q2).reshape(100, 100),
            abs(psi_q) ** 2,
        )

    def test_quadrature_batch(self):
        x1, y1, x2, y2 = 1, 2, -1, -2
        A1, b1, c1 = coherent_state_Abc(x1, y1)
        A2, b2, c2 = coherent_state_Abc(x2, y2)
        A, b, c = math.astensor([A1, A2]), math.astensor([b1, b2]), math.astensor([c1, c2])
        state = Ket.from_bargmann((0,), (A, b, c))
        q = np.linspace(-10, 10, 100)
        psi_q = math.astensor([coherent_state_quad(q, x1, y1), coherent_state_quad(q, x2, y2)]).T
        assert math.allclose(state.quadrature(q), psi_q)
        assert math.allclose(state.quadrature_distribution(q), abs(psi_q) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(q), psi_q)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), abs(psi_q) ** 2)

    @pytest.mark.parametrize("fock", [False, True])
    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_expectation(self, batch_shape, fock):
        alpha_0 = math.broadcast_to(1 + 2j, batch_shape)
        alpha_1 = math.broadcast_to(1 + 3j, batch_shape)

        coh_0 = Coherent(0, x=math.real(alpha_0), y=math.imag(alpha_0))
        coh_1 = Coherent(1, x=math.real(alpha_1), y=math.imag(alpha_1))
        # TODO: clean this up once we have a better way to create batched multimode states
        ket = Ket.from_ansatz((0, 1), coh_0.contract(coh_1, "zip").ansatz)
        ket = ket.to_fock(40) if fock else ket

        # ket operator
        exp_coh_0 = ket.expectation(coh_0)
        exp_coh_1 = ket.expectation(coh_1)
        exp_ket = ket.expectation(ket)

        assert exp_coh_0.shape == batch_shape * 2
        assert exp_coh_1.shape == batch_shape * 2
        assert exp_ket.shape == batch_shape * 2

        assert math.allclose(exp_coh_0, 1)
        assert math.allclose(exp_coh_1, 1)
        assert math.allclose(exp_ket, 1)

        # dm operator
        dm0 = coh_0.dm()
        dm1 = coh_1.dm()
        dm01 = ket.dm()

        exp_dm0 = ket.expectation(dm0)
        exp_dm1 = ket.expectation(dm1)
        exp_dm01 = ket.expectation(dm01)

        assert exp_dm0.shape == batch_shape * 2
        assert exp_dm1.shape == batch_shape * 2
        assert exp_dm01.shape == batch_shape * 2

        assert math.allclose(exp_dm0, 1)
        assert math.allclose(exp_dm1, 1)
        assert math.allclose(exp_dm01, 1)

        # u operator
        beta_0 = 0.1
        beta_1 = 0.2

        u0 = Dgate(0, x=beta_0)
        u1 = Dgate(1, x=beta_1)
        u01 = u0 >> u1

        exp_u0 = ket.expectation(u0)
        exp_u1 = ket.expectation(u1)
        exp_u01 = ket.expectation(u01)

        assert exp_u0.shape == batch_shape
        assert exp_u1.shape == batch_shape
        assert exp_u01.shape == batch_shape

        expected_u0 = math.exp(-(math.abs(beta_0) ** 2) / 2) * math.exp(
            beta_0 * math.conj(alpha_0) - math.conj(beta_0) * alpha_0,
        )
        expected_u1 = math.exp(-(math.abs(beta_1) ** 2) / 2) * math.exp(
            beta_1 * math.conj(alpha_1) - math.conj(beta_1) * alpha_1,
        )

        assert math.allclose(exp_u0, expected_u0)
        assert math.allclose(exp_u1, expected_u1)

        exp_u0_coh = coh_0.expectation(u0)
        exp_u1_coh = coh_1.expectation(u1)

        assert math.allclose(exp_u0, exp_u0_coh)
        assert math.allclose(exp_u1, exp_u1_coh)
        assert math.allclose(exp_u01, exp_u0_coh * exp_u1_coh)

    @pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
    @pytest.mark.parametrize("batch_shape_2", [(7,), (4, 5, 7)])
    def test_expectation_diff_batch_shapes(self, batch_shape, batch_shape_2):
        alpha_0 = math.broadcast_to(1 + 2j, batch_shape)
        coh_0 = Coherent(0, x=math.real(alpha_0), y=math.imag(alpha_0))

        # ket operator
        alpha_1 = math.broadcast_to(0.3 + 0.2j, batch_shape_2)
        coh_1 = Coherent(0, x=math.real(alpha_1), y=math.imag(alpha_1))
        exp_coh_1 = coh_0.expectation(coh_1)
        assert exp_coh_1.shape == batch_shape + batch_shape_2

        # dm operator
        dm1 = coh_1.dm()
        exp_dm1 = coh_0.expectation(dm1)
        assert exp_dm1.shape == batch_shape + batch_shape_2

        # u operator
        beta_0 = math.broadcast_to(0.3, batch_shape_2)
        u0 = Dgate(0, x=beta_0)
        exp_u0 = coh_0.expectation(u0)
        assert exp_u0.shape == batch_shape + batch_shape_2

    def test_expectation_lin_sup(self):
        cat = (Coherent(0, x=1, y=2) + Coherent(0, x=-1, y=2)).normalize()
        assert math.allclose(cat.expectation(cat, mode="zip"), 1.0)
        assert math.allclose(cat.expectation(cat.dm(), mode="zip"), 1.0)
        assert math.allclose(
            cat.expectation(Dgate(0, x=[0.1, 0.2, 0.3])),
            [
                cat.expectation(Dgate(0, x=0.1)),
                cat.expectation(Dgate(0, x=0.2)),
                cat.expectation(Dgate(0, x=0.3)),
            ],
        )

    def test_expectation_error(self):
        ket = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)

        op1 = Attenuator(0)
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            ket.expectation(op1)

        op2 = CircuitComponent._from_attributes(None, Wires(set(), set(), {1}, {0}))
        with pytest.raises(ValueError, match="different modes"):
            ket.expectation(op2)

        op3 = Dgate(2)
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            ket.expectation(op3)

    def test_rshift(self):
        ket = Coherent(0, 1) >> Coherent(1, 1)
        unitary = Dgate(0, 1)
        u_component = CircuitComponent(unitary.ansatz, unitary.wires, unitary.name)
        channel = Attenuator(1, 1)
        ch_component = CircuitComponent(
            channel.ansatz,
            channel.wires,
            channel.name,
        )

        # gates
        assert isinstance(ket >> unitary, Ket)
        assert isinstance(ket >> channel, DM)
        assert isinstance(ket >> unitary >> channel, DM)
        assert isinstance(ket >> channel >> unitary, DM)
        assert isinstance(ket >> u_component, CircuitComponent)
        assert isinstance(ket >> ch_component, CircuitComponent)

        # measurements
        assert isinstance(ket >> Coherent(0, 1).dual, Ket)
        assert isinstance(ket >> Coherent(0, 1).dm().dual, DM)

    @pytest.mark.parametrize("m", [[3], [30], [98], [3, 98]])
    @pytest.mark.parametrize("x", [(0, 1, 2), ([0, 0], [1, 1], [2, 2])])
    def test_get_item(self, m, x):
        x3, x30, x98 = x
        ket = Vacuum((3, 30, 98)) >> Dgate(3, x=x3) >> Dgate(30, x=x30) >> Dgate(98, x=x98)
        dm = ket.dm()
        assert ket[m] == dm[m]

    def test_contract_zip(self):
        coh = Coherent(0, x=[1.0, -1.0])
        displacements = Dgate(0, x=[1.0, -1.0])
        better_cat = coh.contract(displacements, mode="zip")
        assert better_cat == Coherent(0, x=[2.0, -2.0])

    @pytest.mark.parametrize("max_sq", [1, 2, 3])
    def test_random_states(self, max_sq):
        psi = Ket.random((1, 22), max_sq)
        A = psi.ansatz.A
        assert math.allclose(psi.probability, 1)  # checks if the state is normalized
        assert math.allclose(
            A - math.transpose(A),
            math.zeros((2, 2)),
        )  # checks if the A matrix is symmetric

    def test_ipython_repr(self):
        """
        Test the widgets.state function.
        Note: could not mock display because of the states.py file name conflict.
        """
        hbox = state_widget(Number(0, n=1), True, True)
        assert isinstance(hbox, HBox)

        [left, viz_2d] = hbox.children
        assert isinstance(left, VBox)
        assert isinstance(viz_2d, FigureWidget)

        [table, dm] = left.children
        assert isinstance(table, HTML)
        assert isinstance(dm, FigureWidget)

    def test_ipython_repr_too_many_dims(self):
        """Test the widgets.state function when the Ket has too many dims."""
        vbox = state_widget(Vacuum((0, 1)), True, False)
        assert isinstance(vbox, Box)

        [table, wires] = vbox.children
        assert isinstance(table, HTML)
        assert isinstance(wires, HTML)

    def test_is_physical(self):
        assert Ket.random((0, 1)).is_physical
        assert Coherent(0, x=[1, 1, 1]).is_physical

    def test_physical_stellar_decomposition(self):
        r"""
        Tests the physical stellar decomposition.
        """
        # two-mode example:
        psi = Ket.random([0, 1])
        core, U = psi.physical_stellar_decomposition([0])
        assert psi == core >> U

        A_c, _, _ = core.ansatz.triple
        assert math.allclose(A_c[0, 0], 0)

        assert U >> U.dual == Identity([0])

        # many-mode example:
        phi = Ket.random(list(range(5)))
        core, U = phi.physical_stellar_decomposition([0, 2])
        assert phi == core >> U
        assert (core >> Vacuum((1, 3, 4)).dual).normalize() == Vacuum((0, 2))

        A_c, _, _ = core.ansatz.triple
        A_c_reordered = A_c[[0, 2], :]
        A_c_reordered = A_c_reordered[:, [0, 2]]
        assert math.allclose(A_c_reordered, math.zeros((2, 2)))

        # batching test:
        psi = Ket.random([0, 1, 2])
        phi = Ket.random([0, 1, 2])

        sigma = psi + phi
        sigma.ansatz._lin_sup = False
        core, U = sigma.physical_stellar_decomposition([0])
        assert sigma == core.contract(U, mode="zip")

        # displacement test
        phi = Ket.random(list(range(5))) >> Dgate(0, 2) >> Dgate(1, 1)
        core, U = phi.physical_stellar_decomposition([0, 2])
        assert phi == core >> U
        assert (core >> Vacuum((1, 3, 4)).dual).normalize().dm() == Vacuum((0, 2)).dm()

    def test_formal_stellar_decomposition(self):
        psi = Ket.random((0, 1, 2))
        core1, phi1 = psi.formal_stellar_decomposition([1])
        core12, phi12 = psi.formal_stellar_decomposition([1, 2])

        A1, _, _ = phi1.ansatz.triple
        assert math.allclose(A1[1, 1], 0.0)

        A12, _, _ = phi12.ansatz.triple
        assert math.allclose(A12[2:, 2:], math.zeros((2, 2), dtype=math.complex128))

        assert psi == core1 >> phi1
        assert psi == core12 >> phi12
        assert (core12 >> Vacuum(0).dual).normalize() == Vacuum((1, 2))

        psi = Ket.random([0, 1, 2])
        phi = Ket.random([0, 1, 2])

        sigma = psi + phi
        core, U = sigma.formal_stellar_decomposition([0])

        assert sigma == core.contract(U, mode="zip")

    def test_wigner(self):
        ans = Vacuum(0).wigner
        x = np.linspace(0, 1, 100)
        solution = np.exp(-(x**2)) / np.pi

        assert math.allclose(ans(x, 0), solution)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_wigner_poly_exp(self, n):
        psi = (Number(0, n).dm().to_bargmann()) >> Ggate(0)
        xs = np.linspace(-5, 5, 100)
        poly_exp_wig = math.real(psi.wigner(xs, 0))
        wig = wigner_discretized(psi.fock_array(), xs, 0)
        assert math.allclose(poly_exp_wig[:, None], wig[0], atol=3e-3)
