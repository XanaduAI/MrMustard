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

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import numpy as np
import pytest
from ipywidgets import HTML, Box, HBox, VBox
from plotly.graph_objs import FigureWidget

from mrmustard import math, settings

from mrmustard.lab_dev import (
    Attenuator,
    BSgate,
    CircuitComponent,
    Coherent,
    Dgate,
    DM,
    Ket,
    Number,
    QuadratureEigenstate,
    Sgate,
    SqueezedVacuum,
    TraceOut,
    Vacuum,
)
from mrmustard.physics.gaussian import squeezed_vacuum_cov, vacuum_cov, vacuum_means
from mrmustard.physics.representations import Representation
from mrmustard.physics.triples import coherent_state_Abc
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


class TestKet:  # pylint: disable=too-many-public-methods
    r"""
    Tests for the ``Ket`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_ket"])
    @pytest.mark.parametrize("modes", [(0,), (0, 1), (2, 3, 19)])
    def test_init(self, name, modes):
        state = Ket.from_ansatz(modes, None, name)

        assert state.name in ("Ket0", "Ket01", "Ket2319") if not name else name
        assert state.modes == modes
        assert state.wires == Wires(modes_out_ket=set(modes))

    def test_manual_shape(self):
        ket = Coherent(0, x=1)
        assert ket.manual_shape == [None]
        ket.manual_shape[0] = 19
        assert ket.manual_shape == [19]

    def test_auto_shape(self):
        ket = Coherent(0, x=1)
        assert ket.auto_shape() == (8,)
        ket.manual_shape[0] = 19
        assert ket.auto_shape() == (19,)

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
            (modes,), array_in, "my_ket", batch_dims=state_in_fock.ansatz.batch_dims
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
        n_modes = len(modes)

        state1 = Ket.from_phase_space(modes, (vacuum_cov(n_modes), vacuum_means(n_modes), 1.0))
        assert state1 == Vacuum(modes)

        r = [i / 10 for i in range(n_modes)]
        phi = [(i + 1) / 10 for i in range(n_modes)]
        state2 = Ket.from_phase_space(
            modes, (squeezed_vacuum_cov(r, phi), vacuum_means(n_modes), 1.0)
        )
        exp_state = Vacuum(modes)
        for mode, r_i, phi_i in zip(modes, r, phi):
            exp_state = exp_state >> Sgate(mode, r_i, phi_i)
        assert state2 == exp_state

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
            state.to_fock(40).quadrature_distribution(q, phi=phi), abs(psi_phi) ** 2
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
            state.quadrature_distribution(q1, q2).reshape(100, 100), abs(psi_q) ** 2
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

    def test_expectation_bargmann(self):
        ket = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)

        assert math.allclose(ket.expectation(ket), 1.0)

        k0 = Coherent(0, x=1, y=2)
        k1 = Coherent(1, x=1, y=3)
        k01 = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)

        res_k0 = (ket.contract(k0.dual)) >> TraceOut(1)
        res_k1 = (ket.contract(k1.dual)) >> TraceOut(0)
        res_k01 = ket.contract(k01.dual)

        assert math.allclose(ket.expectation(k0), res_k0)
        assert math.allclose(ket.expectation(k1), res_k1)
        assert math.allclose(ket.expectation(k01), math.sum(res_k01.ansatz.c))

        dm0 = Coherent(0, x=1, y=2).dm()
        dm1 = Coherent(1, x=1, y=3).dm()
        dm01 = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).dm()

        res_dm0 = (ket.contract(ket.adjoint).contract(dm0.dual)) >> TraceOut(1)
        res_dm1 = (ket.contract(ket.adjoint).contract(dm1.dual)) >> TraceOut(0)
        res_dm01 = ket.contract(ket.adjoint).contract(dm01.dual)

        assert math.allclose(ket.expectation(dm0), res_dm0)
        assert math.allclose(ket.expectation(dm1), res_dm1)
        assert math.allclose(ket.expectation(dm01), math.sum(res_dm01.ansatz.c))

        u0 = Dgate(0, x=0.1)
        u1 = Dgate(1, x=0.2)
        u01 = Dgate(0, x=0.3) >> Dgate(1, x=0.4)

        res_u0 = (ket.contract(u0)) >> ket.dual
        res_u1 = (ket.contract(u1)) >> ket.dual
        res_u01 = (ket.contract(u01)) >> ket.dual

        assert math.allclose(ket.expectation(u0), res_u0)
        assert math.allclose(ket.expectation(u1), res_u1)
        assert math.allclose(ket.expectation(u01), res_u01)

    def test_expectation_fock(self):
        ket = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).to_fock(10)

        assert math.allclose(ket.expectation(ket), math.abs(ket >> ket.dual) ** 2)
        k0 = Coherent(0, x=1, y=2).to_fock(10)
        k1 = Coherent(1, x=1, y=3).to_fock(10)
        k01 = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).to_fock(10)

        res_k0 = (ket.contract(k0.dual)) >> TraceOut(1)
        res_k1 = (ket.contract(k1.dual)) >> TraceOut(0)
        res_k01 = (ket >> k01.dual) ** 2

        assert math.allclose(ket.expectation(k0), res_k0)
        assert math.allclose(ket.expectation(k1), res_k1)
        assert math.allclose(ket.expectation(k01), res_k01)

        dm0 = Coherent(0, x=1, y=0.2).dm().to_fock(10)
        dm1 = Coherent(1, x=1, y=0.3).dm().to_fock(10)
        dm01 = (Coherent(0, x=1, y=0.2) >> Coherent(1, x=1, y=0.3)).dm().to_fock(10)

        res_dm0 = (ket.contract(ket.adjoint).contract(dm0.dual)) >> TraceOut(1)
        res_dm1 = (ket.contract(ket.adjoint).contract(dm1.dual)) >> TraceOut(0)
        res_dm01 = (ket.contract(ket.adjoint).contract(dm01.dual)).to_fock(10).ansatz.array

        assert math.allclose(ket.expectation(dm0), res_dm0)
        assert math.allclose(ket.expectation(dm1), res_dm1)
        assert math.allclose(ket.expectation(dm01), res_dm01)

        u0 = Dgate(1, x=0.1)
        u1 = Dgate(0, x=0.2)
        u01 = Dgate(0, x=0.3) >> Dgate(1, x=0.4)

        res_u0 = (ket.contract(u0).contract(ket.dual)).to_fock(10).ansatz.array
        res_u1 = (ket.contract(u1).contract(ket.dual)).to_fock(10).ansatz.array
        res_u01 = (ket.contract(u01).contract(ket.dual)).to_fock(10).ansatz.array

        assert math.allclose(ket.expectation(u0), res_u0)
        assert math.allclose(ket.expectation(u1), res_u1)
        assert math.allclose(ket.expectation(u01), res_u01)

    def test_expectation_error(self):
        ket = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)

        op1 = Attenuator(0)
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            ket.expectation(op1)

        op2 = CircuitComponent(Representation(wires=Wires(set(), set(), {1}, {0})))
        with pytest.raises(ValueError, match="different modes"):
            ket.expectation(op2)

        op3 = Dgate(2)
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            ket.expectation(op3)

    def test_rshift(self):
        ket = Coherent(0, 1) >> Coherent(1, 1)
        unitary = Dgate(0, 1)
        u_component = CircuitComponent(unitary.representation, unitary.name)
        channel = Attenuator(1, 1)
        ch_component = CircuitComponent(
            channel.representation,
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
            A - math.transpose(A), math.zeros((2, 2))
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
