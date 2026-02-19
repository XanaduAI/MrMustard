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

"""Tests for the density matrix."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import (
    DM,
    Attenuator,
    CircuitComponent,
    Coherent,
    Dgate,
    GaussianDM,
    GaussianKet,
    Ket,
    Number,
    SqueezedVacuum,
    Vacuum,
)
from mrmustard.physics.triples import coherent_state_Abc
from mrmustard.physics.utils import random_Abc
from mrmustard.physics.wires import Wires


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


class TestDM:
    r"""
    Tests for the ``DM`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_dm"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {2, 3, 19}])
    def test_init(self, name, modes):
        state = DM.from_ansatz(modes, None, name)

        assert name if name else state.name in ("DM0", "DM01", "DM2319")
        assert list(state.modes) == sorted(modes)
        assert state.wires == Wires(modes_out_bra=modes, modes_out_ket=modes)

    def test_is_separable(self):
        entangled_state = GaussianDM.random(modes=(0, 1, 2))
        assert not entangled_state.is_separable

        separable_state = (
            GaussianDM.random(modes=0) >> GaussianDM.random(modes=1) >> GaussianDM.random(modes=2)
        )
        assert separable_state.is_separable

        entangled_state = GaussianDM.random(modes=0) >> GaussianDM.random(modes=(1, 2))
        assert not entangled_state.is_separable

    def test_manual_shape(self):
        dm = Coherent(0, alpha=1).dm()
        assert dm.manual_shape == (None, None)
        dm.manual_shape = (19, None)
        assert dm.manual_shape == (19, None)

    def test_auto_shape(self):
        dm = Coherent(0, 1).dm()
        assert dm.auto_shape() == (8, 8)
        dm.manual_shape = (1, 8)
        assert dm.auto_shape() == (1, 8)

        dm = Coherent(0, 1).dm() >> Number(1, 10).dual
        assert dm.auto_shape() == (8, 11, 8, 11)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_to_from_bargmann(self, modes, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state_in = Coherent(modes, x + 1j * y) >> Attenuator(modes, 0.7)
        triple_in = state_in.bargmann_triple()

        state_out = DM.from_bargmann((modes,), triple_in, "my_dm")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = (Coherent(0, 1) >> Coherent(1, 1)).dm()
        with pytest.raises(ValueError):
            DM.from_bargmann(
                (0,),
                state01.bargmann_triple(),
                "my_dm",
            )

    def test_from_fock_error(self):
        state01 = (Coherent(0, 1) >> Coherent(1, 1)).dm()
        state01 = state01.to_fock(2)
        with pytest.raises(ValueError):
            DM.from_fock((0,), state01.fock_array(5), "my_dm")

    def test_bargmann_triple_error(self):
        fock = Number(0, n=10).dm()
        with pytest.raises(AttributeError):
            fock.bargmann_triple()

    @pytest.mark.parametrize("coeff", [0.5, 0.3])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_normalize(self, coeff, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x + 1j * y).dm()
        state *= coeff
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)
        state = state.to_fock(5)  # truncated
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("coeff", [0.5, 0.3])
    def test_normalize_mixture(self, coeff):
        state = Coherent(0, 1 + 1j).dm() + Coherent(0, -1 - 1j).dm()
        state *= coeff
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)
        fock_state = state.to_fock(5)
        normalized = fock_state.normalize()
        assert math.allclose(normalized.probability, 1.0)

        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        lin_sup_state_batch = DM.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        normalized_batch = lin_sup_state_batch.normalize()
        assert math.allclose(normalized_batch.probability, 1.0)
        fock_state = state.to_fock(5)
        normalized = fock_state.normalize()
        assert math.allclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_to_from_fock(self, modes, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state_in = Coherent(modes, x + 1j * y) >> Attenuator(modes, 0.8)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock_array(5)

        assert math.allclose(array_in, state_in_fock.ansatz.array)

        state_out = DM.from_fock(
            (modes,),
            array_in,
            "my_dm",
            batch_dims=state_in_fock.ansatz.batch_dims,
        )
        assert state_in_fock == state_out

    def test_to_from_fock_mixture(self):
        state_in = Coherent(0, 1 + 2j).dm() + Coherent(0, -1 - 2j).dm()
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock_array(5)

        assert math.allclose(array_in, state_in_fock.ansatz.array)

        state_out = DM.from_fock((0,), array_in, "my_dm")
        assert state_in_fock == state_out

        A, b, c = state_in.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        lin_sup_state_batch = DM.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        state_in_fock = lin_sup_state_batch.to_fock(5)
        array_in = lin_sup_state_batch.fock_array(5)

        assert math.allclose(array_in, state_in_fock.ansatz.array)

        state_out = DM.from_fock(
            (0,),
            array_in,
            "my_dm",
            batch_dims=lin_sup_state_batch.ansatz.batch_dims - 1,
        )
        assert state_in_fock == state_out

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_phase_space(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state0 = Coherent(0, x + 1j * y) >> Attenuator(0, 1.0)
        cov, means, coeff = state0.phase_space(s=0)
        assert cov.shape[:-2] == state0.ansatz.batch_shape
        assert means.shape[:-1] == state0.ansatz.batch_shape
        assert coeff.shape == state0.ansatz.batch_shape
        assert math.allclose(coeff, 1.0)
        assert math.allclose(cov, math.eye(2) * settings.HBAR / 2)
        assert math.allclose(means, math.astensor([1.0, 2.0]) * math.sqrt(settings.HBAR * 2))

    def test_from_phase_space(self):
        state = Coherent(0, 1 + 2j) >> Attenuator(0, 0.8)
        cov, means, coeff = state.phase_space(s=0)
        state1 = DM.from_phase_space([0], (cov, means, coeff))
        assert state1 == state
        with pytest.raises(ValueError):
            DM.from_phase_space((0, 1), (cov, means, coeff))

    @pytest.mark.parametrize("modes", [(0,), (0, 1), (2, 3, 19)])
    @pytest.mark.parametrize("batch_shape", [(1,), (2, 3)])
    def test_to_from_quadrature(self, modes, batch_shape):
        A, b, c = random_Abc(len(modes) * 2, batch_shape)
        state0 = DM.from_bargmann(modes, (A, b, c))
        Atest, btest, ctest = state0.quadrature_triple()
        state1 = DM.from_quadrature(modes, (Atest, btest, ctest))
        Atest2, btest2, ctest2 = state1.bargmann_triple()
        assert math.allclose(Atest2, A)
        assert math.allclose(btest2, b)
        assert math.allclose(ctest2, c)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_L2_norm(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x + 1j * y).dm()
        L2_norm = state.L2_norm
        assert L2_norm.shape == state.ansatz.batch_shape
        assert math.allclose(L2_norm, 1)

    def test_L2_norm_mixture(self):
        state = Coherent(0, 1).dm() + Coherent(0, -1).dm()
        L2_norm = state.L2_norm
        assert L2_norm.shape == state.ansatz.batch_shape[:-1]
        assert math.allclose(L2_norm, 2.03663128)

        A, b, c = state.ansatz.triple
        A_batch = math.astensor([A, A, A])
        b_batch = math.astensor([b, b, b])
        c_batch = math.astensor([c, c, c])
        state_batch = DM.from_bargmann((0,), (A_batch, b_batch, c_batch), lin_sup=True)
        L2_norm = state_batch.L2_norm
        assert L2_norm.shape == state_batch.ansatz.batch_shape[:-1]
        assert math.allclose(L2_norm, 2.03663128)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_probability(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(0, x + 1j * y).dm()
        prob = state.probability
        assert prob.shape == state.ansatz.batch_shape
        assert math.allclose(prob, 1)
        assert math.allclose(state.to_fock(20).probability, 1)

    def test_probability_mixture(self):
        state = (Coherent(0, 1).dm() + Coherent(0, -1).dm()).normalize()
        prob = state.probability
        assert prob.shape == state.ansatz.batch_shape[:-1]
        assert math.allclose(prob, 1)
        assert math.allclose(state.to_fock(20).probability, 1)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_purity(self, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(mode=0, alpha=x + 1j * y).dm()
        assert math.allclose(state.purity, 1)
        assert state.is_pure

    def test_purity_mixture(self):
        state = Coherent(0, 1).dm() + Coherent(0, -1).dm()
        assert math.allclose(state.purity, state.normalize().purity)
        assert not state.is_pure

    def test_quadrature_single_mode_dm(self):
        x, y = 1, 2
        state = Coherent(mode=0, alpha=x + 1j * y).dm()
        q = np.linspace(-10, 10, 100)
        quad0 = q
        quad1 = q + 1
        ket = coherent_state_quad(q + 1, x, y)
        bra = math.conj(coherent_state_quad(q, x, y))
        assert math.allclose(state.quadrature(quad0, quad1), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(quad0, quad1), bra * ket)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra) ** 2)

    def test_quadrature_multimode_dm(self):
        x, y = 1, 2
        state = (Coherent(mode=0, alpha=x + 1j * y) >> Coherent(mode=1, alpha=x + 1j * y)).dm()
        q = np.linspace(-10, 10, 100)
        ket = math.kron(coherent_state_quad(q, x, y), coherent_state_quad(q, x, y))
        bra = math.kron(
            np.conj(coherent_state_quad(q, x, y)),
            np.conj(coherent_state_quad(q, x, y)),
        )
        assert math.allclose(state.quadrature(q, q, q, q), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)

        ket_slice = math.kron(coherent_state_quad(q + 1, x, y), coherent_state_quad(q + 1, x, y))
        bra_slice = math.kron(
            np.conj(coherent_state_quad(q, x, y)),
            np.conj(coherent_state_quad(q, x, y)),
        )

        assert math.allclose(
            state.to_fock(40).quadrature(q, q, q + 1, q + 1),
            bra_slice * ket_slice,
        )
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra_slice) ** 2)

    def test_quadrature_multivariable_dm(self):
        x, y = 1, 2
        state = Coherent(mode=0, alpha=x + 1j * y).dm() >> Coherent(mode=1, alpha=x + 1j * y).dm()
        q1 = np.linspace(-10, 10, 100)
        q2 = np.linspace(-10, 10, 100)
        psi_q = math.outer(coherent_state_quad(q1, x, y), coherent_state_quad(q2, x, y))
        assert math.allclose(
            state.quadrature_distribution(q1, q2).reshape(100, 100),
            abs(psi_q) ** 2,
        )

    def test_quadrature_batch(self):
        x1, y1, x2, y2 = 1, 2, -1, -2
        A1, b1, c1 = coherent_state_Abc(x1 + 1j * y1)
        A2, b2, c2 = coherent_state_Abc(x2 + 1j * y2)
        A, b, c = math.astensor([A1, A2]), math.astensor([b1, b2]), math.astensor([c1, c2])
        state = Ket.from_bargmann((0,), (A, b, c)).dm()
        q = np.linspace(-10, 10, 100)

        ket = math.astensor([coherent_state_quad(q, x1, y1), coherent_state_quad(q, x2, y2)]).T
        bra = math.astensor(
            [np.conj(coherent_state_quad(q, x1, y1)), np.conj(coherent_state_quad(q, x2, y2))],
        ).T

        assert math.allclose(state.quadrature(q, q), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(q, q), bra * ket)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra) ** 2)

    @pytest.mark.parametrize("fock", [False, True])
    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_expectation_ket(self, batch_shape, fock):
        alpha_0 = math.broadcast_to(1 + 2j, batch_shape)
        alpha_1 = math.broadcast_to(1 + 3j, batch_shape)
        coh_0 = Coherent(0, alpha=alpha_0)
        coh_1 = Coherent(1, alpha=alpha_1)
        ket = Ket.from_ansatz((0, 1), coh_0.contract(coh_1).ansatz)
        ket = ket.to_fock(40) if fock else ket
        dm = ket.dm()

        # ket operator
        exp_coh_0 = dm.expectation(coh_0)
        exp_coh_1 = dm.expectation(coh_1[None, ...])
        exp_ket = dm.expectation(ket)

        assert exp_coh_0.shape == batch_shape
        assert exp_coh_1.shape == (1, *batch_shape)
        assert exp_ket.shape == batch_shape

        assert math.allclose(exp_coh_0, 1)
        assert math.allclose(exp_coh_1, 1)
        assert math.allclose(exp_ket, 1)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_expectation_dm(self, batch_shape):
        alpha_0 = math.broadcast_to(1 + 2j, batch_shape)
        alpha_1 = math.broadcast_to(1 + 3j, batch_shape)
        coh_0 = Coherent(0, alpha=alpha_0)
        coh_1 = Coherent(1, alpha=alpha_1)
        coh_01 = coh_0.contract(coh_1)
        dm = Ket.from_ansatz((0, 1), coh_01.ansatz).dm()

        dm0 = coh_0.dm()
        dm1 = coh_1.dm()

        exp_dm0 = dm.expectation(dm0)
        exp_dm1 = dm.expectation(dm1[None, ...])
        exp_dm01 = dm.expectation(dm)

        assert exp_dm0.shape == batch_shape
        assert exp_dm1.shape == (1, *batch_shape)
        assert exp_dm01.shape == batch_shape

        assert math.allclose(exp_dm0, 1)
        assert math.allclose(exp_dm1, 1)
        assert math.allclose(exp_dm01, 1)

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_expectation_u(self, batch_shape):
        alpha_0 = math.broadcast_to(1 + 2j, batch_shape)
        alpha_1 = math.broadcast_to(1 + 3j, batch_shape)
        coh_0 = Coherent(0, alpha=alpha_0)
        coh_1 = Coherent(1, alpha=alpha_1)
        coh_01 = coh_0.contract(coh_1)
        dm_0 = Ket.from_ansatz((0, 1), coh_01.ansatz).dm()

        beta_0 = 0.1
        beta_1 = 0.2

        u0 = Dgate(0, alpha=beta_0)
        u1 = Dgate(1, alpha=beta_1)
        u01 = u0 >> u1

        exp_u0 = dm_0.expectation(u0)
        exp_u1 = dm_0[None, ...].expectation(u1[None])
        exp_u01 = dm_0.expectation(u01)

        assert exp_u0.shape == batch_shape
        assert exp_u1.shape == (1, *batch_shape)
        assert exp_u01.shape == batch_shape

        expected_u0 = math.exp(-(math.abs(beta_0) ** 2) / 2) * math.exp(
            beta_0 * math.conj(alpha_0) - math.conj(beta_0) * alpha_0,
        )
        expected_u1 = math.exp(-(math.abs(beta_1) ** 2) / 2) * math.exp(
            beta_1 * math.conj(alpha_1) - math.conj(beta_1) * alpha_1,
        )

        assert math.allclose(exp_u0, expected_u0)
        assert math.allclose(exp_u1, expected_u1)

        exp_u0_coh_0 = coh_0.expectation(u0)
        exp_u1_coh_1 = coh_1.expectation(u1[None])

        assert math.allclose(exp_u0, exp_u0_coh_0)
        assert math.allclose(exp_u1, exp_u1_coh_1)
        assert math.allclose(exp_u01, exp_u0_coh_0 * exp_u1_coh_1)

    def test_expectation_with_different_batch_shapes(self):
        # init state
        coh2 = Coherent(0, alpha=math.broadcast_to(0.3 + 0.2j, (2,)))
        coh7 = Coherent(0, alpha=math.broadcast_to(0.3 + 0.2j, (7,)))
        dm2 = coh2.dm()

        # expectation with ket
        assert dm2.expectation(coh2).shape == (2,)
        assert dm2[:, None].expectation(coh7[None, :]).shape == (2, 7)

        # expectation with dm
        assert dm2.expectation(coh2.dm()).shape == (2,)
        assert dm2[:, None].expectation(coh7.dm()[None, :]).shape == (2, 7)

        # u operator
        u0 = Dgate(0, alpha=math.broadcast_to(0.3, (2,)))
        assert dm2.expectation(u0).shape == (2,)
        u0 = Dgate(0, alpha=math.broadcast_to(0.3, (7,)))
        assert dm2[:, None].expectation(u0[None, :]).shape == (2, 7)

    @pytest.mark.parametrize(
        "operator",
        [Coherent(0, alpha=0.1 + 0.2j).dm(), Dgate(0, alpha=-0.1 - 0.2j)],
        ids=["DM_LIKE", "UNITARY_LIKE"],
    )
    def test_expectation_shape_handling(self, operator):
        dm = SqueezedVacuum(0, 0.5).dm()
        shape = 20
        dm.manual_shape = (shape, shape)  # otherwise autoshape too small when combined with dgate

        expectation_barg_fock = math.abs(dm.to_bargmann().expectation(operator.to_fock(shape)))
        expectation_fock_barg = math.abs(dm.to_fock(shape).expectation(operator.to_bargmann()))
        expectation_fock_fock = math.abs(dm.to_fock(shape).expectation(operator.to_fock(shape)))

        assert math.allclose(expectation_barg_fock, expectation_fock_barg)
        assert math.allclose(expectation_fock_fock, expectation_barg_fock)
        assert math.allclose(expectation_fock_fock, expectation_fock_barg)

    def test_expectation_lin_sup(self):
        cat = (Coherent(0, alpha=0.1 + 0.2j) + Coherent(0, alpha=-0.1 - 0.2j)).normalize()
        cat_dm = cat.dm()
        assert math.allclose(cat_dm.expectation(cat), 1.0)
        assert math.allclose(cat_dm.expectation(cat_dm), 1.0)
        assert math.allclose(
            cat_dm.expectation(Dgate(0, alpha=[0.1, 0.2, 0.3])),
            [
                cat_dm.expectation(Dgate(0, alpha=0.1)),
                cat_dm.expectation(Dgate(0, alpha=0.2)),
                cat_dm.expectation(Dgate(0, alpha=0.3)),
            ],
        )

    def test_expectation_error(self):
        dm = GaussianDM.random(modes=(0, 1))

        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            dm.expectation(Attenuator(0))

        with pytest.raises(ValueError, match="No modes in common"):
            dm.expectation(GaussianDM.random(modes=(2,)))

        with pytest.raises(ValueError, match="No modes in common"):
            dm.expectation(Dgate(2))

    @pytest.mark.parametrize("n_modes", [1, 2, 3])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_fock_distribution(self, n_modes, batch_shape):
        cutoff = 5
        state = GaussianKet.random(tuple(range(n_modes)))
        A, b, c = state.ansatz.triple
        A_batch = math.broadcast_to(A, batch_shape + A.shape)
        b_batch = math.broadcast_to(b, batch_shape + b.shape)
        c_batch = math.broadcast_to(c, batch_shape + c.shape)
        state = Ket.from_bargmann(state.modes, (A_batch, b_batch, c_batch))
        fock_dist = state.fock_distribution(cutoff)
        fock_dist_dm = state.dm().fock_distribution(cutoff)
        assert math.allclose(fock_dist, fock_dist_dm)
        assert fock_dist.shape == (*batch_shape, (cutoff + 1) ** n_modes)

        alpha = math.broadcast_to(1 + 0.1j, batch_shape)
        state_separable = Coherent(0, alpha=alpha)
        for i in range(n_modes - 1):
            state_separable >>= Coherent(i + 1, alpha=alpha)
        fock_dist_separable = state_separable.fock_distribution(cutoff)
        fock_dist_dm_separable = state_separable.dm().fock_distribution(cutoff)
        assert math.allclose(fock_dist_separable, fock_dist_dm_separable)
        assert fock_dist_separable.shape == (*batch_shape, (cutoff + 1) ** n_modes)

    def test_rshift(self):
        ket = Coherent(0, 1) >> Coherent(1, 1)

        unitary = Dgate(0, 1)
        u_component = CircuitComponent(
            ansatz_factory=unitary.ansatz_factory,
            wires=unitary.wires,
            name=unitary.name,
        )
        u_component.parameters["alpha"] = unitary.parameters.alpha

        channel = Attenuator(1, 1)
        ch_component = CircuitComponent(
            ansatz_factory=channel.ansatz_factory,
            wires=channel.wires,
            name=channel.name,
        )
        ch_component.parameters["transmissivity"] = channel.parameters.transmissivity
        dm = ket >> channel

        # gates
        assert isinstance(dm, DM)
        assert isinstance(dm >> unitary >> channel, DM)
        assert isinstance(dm >> channel >> unitary, DM)
        assert isinstance(dm >> u_component, CircuitComponent)
        assert isinstance(dm >> ch_component, CircuitComponent)

        # measurements
        assert isinstance(dm >> Coherent(0, 1).dual, DM)
        assert isinstance(dm >> Coherent(0, 1).dm().dual, DM)

    @pytest.mark.parametrize("modes", [(5,), (1, 2)])
    def test_random(self, modes):
        m = len(modes)
        dm = GaussianDM.random(modes)
        A = dm.ansatz.A
        Gamma = A[..., :m, m:]
        Lambda = A[..., m:, m:]
        Temp = Gamma + math.conj(Lambda.T) @ math.inv(1 - Gamma.T) @ Lambda
        assert np.all(
            np.linalg.eigvals(Gamma) >= 0,
        )  # checks if the off-diagonal block of dm is PSD
        assert np.all(np.linalg.eigvals(Gamma) < 1)
        assert np.all(np.linalg.eigvals(Temp) < 1)

    def test_random_seed(self):
        # same seed should produce same state
        assert GaussianDM.random(modes=[0, 1], seed=42) == GaussianDM.random(modes=[0, 1], seed=42)
        # different seeds should produce different states
        assert GaussianDM.random(modes=[0, 1], seed=42) != GaussianDM.random(modes=[0, 1], seed=43)

        # local seed should not affect global seed
        settings.SEED = 42
        dm_from_global_1 = GaussianDM.random(modes=[0, 1])
        dm_from_global_2 = GaussianDM.random(modes=[0, 1])

        settings.SEED = 42
        dm_from_global_1_redux = GaussianDM.random(modes=[0, 1])
        # this call should not affect the global RNG
        _ = GaussianDM.random(modes=[0, 1], seed=123)
        dm_from_global_2_redux = GaussianDM.random(modes=[0, 1])

        assert dm_from_global_1 == dm_from_global_1_redux
        assert dm_from_global_2 == dm_from_global_2_redux

        # no modes should raise error
        with pytest.raises(ValueError, match="Cannot create a random GaussianDM with no modes."):
            GaussianDM.random(modes=[])

    def test_is_positive(self):
        assert (GaussianKet.random((2, 9)) >> Attenuator(2) >> Attenuator(9)).is_positive
        assert (
            Coherent(2, alpha=[1 + 2j, 1 + 2j, 1 + 2j])
            >> Coherent(9, alpha=[1 + 2j, 1 + 2j, 1 + 2j])
            >> Attenuator(2)
            >> Attenuator(9)
        ).is_positive
        A = np.zeros((2, 4, 4))
        A[..., 0, -1] = 1.0
        rho = DM.from_bargmann(
            (2, 9),
            [A, math.zeros((2, 4), dtype=math.complex128), math.ones(2, dtype=math.complex128)],
        )  # this test fails at the hermitian check
        assert not rho.is_positive

    @pytest.mark.parametrize("modes", [tuple(range(10)), (0, 1)])
    def test_is_physical(self, modes):
        rho = GaussianDM.random(modes)
        assert rho.is_physical
        rho = 2 * rho
        assert not rho.is_physical
        assert GaussianKet.random(modes).dm().is_physical
        assert Coherent(0, alpha=[1, 1, 1]).dm().is_physical

    def test_fock_array_ordering(self):
        rho = (Number(0, 0) + 1j * Number(0, 1)).dm()
        rho_fock = rho.fock_array(standard_order=True)

        assert math.allclose(
            rho_fock,
            math.astensor([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]]),
        )
        assert math.allclose(
            Coherent(0, alpha=1).dm().fock_array(8, standard_order=True),
            Coherent(0, alpha=[1, 1, 1]).dm().fock_array(8, standard_order=True),
        )

    def test_formal_stellar_decomposition(self):
        rho = GaussianDM.random([0, 1])
        sigma, phi = rho.formal_stellar_decomposition([0])

        assert sigma.modes == (0, 1)
        assert phi.modes == (0,)

        # testing the validness of contraction equality
        test_A, test_b, test_c = (sigma >> phi).ansatz.triple
        A, b, c = rho.ansatz.triple

        assert math.allclose(A, test_A)
        assert math.allclose(b, test_b)
        assert math.allclose(c, test_c)

        # testing the core conditions on sigma
        As, _, _ = sigma.ansatz.triple
        assert As[0, 0] == 0 and As[0, 2] == 0 and As[2, 2] == 0

        # 4-mode example
        rho = GaussianDM.random([0, 1, 2, 3])
        core, phi = rho.formal_stellar_decomposition([0, 3])

        # displacement test
        rho = GaussianDM.random([0, 1]) >> Dgate(0, 0.5) >> Dgate(1, 0.2)
        core, phi = rho.formal_stellar_decomposition([0])
        assert (core >> Vacuum(1).dual).normalize() == Vacuum((0,)).dm()
        assert rho == core >> phi

        # batch test
        rho1 = GaussianDM.random([0, 1, 2, 3]) >> Dgate(0, 0.7)
        rho2 = GaussianDM.random([0, 1, 2, 3])

        sigma = rho1 + rho2
        sigma.ansatz._lin_sup = False
        core, phi = sigma.formal_stellar_decomposition([0, 1])

        assert core.contract(phi) == sigma

    def test_physical_stellar_decomposition(self):
        rho = GaussianDM.random([0, 1])
        core, phi = rho.physical_stellar_decomposition([0])

        assert rho == core >> phi
        assert core.is_physical
        assert isinstance(core, Ket)

        A = core.ansatz.A
        assert math.allclose(A[0, 0], 0)

        # 4-mode example
        rho = GaussianDM.random([0, 1, 2, 3])
        core, phi = rho.physical_stellar_decomposition([0, 3])

        assert rho == core >> phi
        assert phi.is_physical
        assert (core >> Vacuum((1, 2)).dual).normalize() == Vacuum((0, 3))

        # testing displacement
        rho = GaussianDM.random([0, 1]) >> Dgate(0, 0.5)
        core, phi = rho.physical_stellar_decomposition([0])
        assert (core >> Vacuum(1).dual).normalize() == Vacuum((0,))

        # testing batches:
        rho1 = GaussianDM.random([0, 1, 2, 3]) >> Dgate(0, 1)
        rho2 = GaussianDM.random([0, 1, 2, 3])

        sigma = rho1 + rho2
        sigma.ansatz._lin_sup = False
        core, phi = sigma.physical_stellar_decomposition([0, 1])

        assert core.dm().contract(phi) == sigma

    def test_stellar_decomposition_mixed(self):
        """Test DM stellar decomposition for cases where M > n/2 (returns DM core)."""
        # 4-mode example with M=3 > n/2=2: returns DM
        rho = GaussianDM.random([0, 1, 2, 3])
        core, phi = rho.physical_stellar_decomposition([0, 1, 2])

        assert rho == core >> phi
        assert core.is_physical
        assert isinstance(core, DM)

        # Check core property on core modes
        A = core.ansatz.A
        assert math.allclose(A[:3, :3], math.zeros((3, 3)))

        # 6-mode example with M=4 > n/2=3: returns DM
        rho = GaussianDM.random([0, 1, 2, 3, 4, 5])
        core, phi = rho.physical_stellar_decomposition([0, 1, 2, 3])

        assert rho == core >> phi
        assert isinstance(core, DM)

        # batched displaced example with M=3 > n/2=2: returns DM
        rho1 = GaussianDM.random([0, 1, 2, 3]) >> Dgate(0, 1)
        rho2 = GaussianDM.random([0, 1, 2, 3])

        sigma = rho1 + rho2
        sigma.ansatz._lin_sup = False
        core, phi = sigma.physical_stellar_decomposition([0, 1, 2])

        assert isinstance(core, DM)
        assert math.allclose(core.dm().contract(phi).ansatz.A, sigma.ansatz.A)
        assert math.allclose(core.dm().contract(phi).ansatz.b, sigma.ansatz.b)

    def test_wigner(self):
        ans = Vacuum(0).dm().wigner
        x = np.linspace(0, 1, 100)

        solution = np.exp(-(x**2)) / np.pi

        assert math.allclose(ans(x, 0), solution)

    def test_wormhole_1mode_single_outcome(self):
        """Test wormhole_1mode with a single PNR outcome."""
        dm = GaussianDM.random([0, 1], seed=42)
        cutoff = 10
        pnr = 3

        # Get conditional DM using wormhole (tuple API) - returns dict
        results = dm.wormhole_1mode((pnr,), output_cutoff=cutoff, leftover_mode=0)
        assert isinstance(results, dict)
        cond_dm = results[(pnr,)]

        # Verify it's a single-mode DM
        assert isinstance(cond_dm, DM)
        assert cond_dm.modes == (0,)
        assert cond_dm.fock_array().shape == (cutoff + 1, cutoff + 1)

        # Verify against full tensor slice
        A, b, c = dm.bargmann_triple()
        full_shape = (cutoff + 1, pnr + 1, cutoff + 1, pnr + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_array = np.array(full_tensor[:, pnr, :, pnr])

        assert math.allclose(cond_dm.fock_array(), ref_array, atol=1e-10)

    def test_wormhole_1mode_multiple_outcomes(self):
        """Test wormhole_1mode with multiple PNR outcomes."""
        dm = GaussianDM.random([0, 1], seed=123)
        cutoff = 8
        pnr_list = [(0,), (1,), (2,)]

        results = dm.wormhole_1mode(pnr_list, output_cutoff=cutoff, leftover_mode=0)

        # Verify we get a dict with the right keys
        assert isinstance(results, dict)
        assert (0,) in results
        assert (1,) in results
        assert (2,) in results

        # Verify each result is correct
        A, b, c = dm.bargmann_triple()
        for pnr_tuple in pnr_list:
            pnr = pnr_tuple[0]
            full_shape = (cutoff + 1, pnr + 1, cutoff + 1, pnr + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_array = np.array(full_tensor[:, pnr, :, pnr])

            assert math.allclose(results[pnr_tuple].fock_array(), ref_array, atol=1e-10)

    def test_wormhole_1mode_3modes(self):
        """Test wormhole_1mode with a 3-mode DM."""
        dm = GaussianDM.random([0, 1, 2], seed=456)
        cutoff = 6
        pnr1, pnr2 = 2, 3

        # Tuple specifies PNR for modes 1 and 2 (in order), leftover is mode 0
        results = dm.wormhole_1mode((pnr1, pnr2), output_cutoff=cutoff, leftover_mode=0)
        cond_dm = results[(pnr1, pnr2)]

        # Verify against full tensor
        A, b, c = dm.bargmann_triple()
        full_shape = (cutoff + 1, pnr1 + 1, pnr2 + 1, cutoff + 1, pnr1 + 1, pnr2 + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_array = np.array(full_tensor[:, pnr1, pnr2, :, pnr1, pnr2])

        assert math.allclose(cond_dm.fock_array(), ref_array, atol=1e-10)

    def test_wormhole_1mode_different_leftover(self):
        """Test wormhole_1mode with different leftover modes."""
        dm = GaussianDM.random([0, 1], seed=789)
        cutoff = 8
        pnr = 2

        # Leftover mode 0
        results_0 = dm.wormhole_1mode((pnr,), output_cutoff=cutoff, leftover_mode=0)
        cond_dm_0 = results_0[(pnr,)]
        assert cond_dm_0.modes == (0,)

        # Leftover mode 1
        results_1 = dm.wormhole_1mode((pnr,), output_cutoff=cutoff, leftover_mode=1)
        cond_dm_1 = results_1[(pnr,)]
        assert cond_dm_1.modes == (1,)

    def test_wormhole_1mode_errors(self):
        """Test wormhole_1mode raises appropriate errors."""
        dm = GaussianDM.random([0, 1], seed=111)

        # Wrong leftover mode
        with pytest.raises(ValueError, match="leftover_mode 5 is not in"):
            dm.wormhole_1mode((0,), output_cutoff=5, leftover_mode=5)

        # Wrong tuple length
        with pytest.raises(ValueError, match="must have length"):
            dm.wormhole_1mode((0, 0), output_cutoff=5, leftover_mode=0)

        # Single mode DM
        dm_single = Coherent(0, 1).dm()
        with pytest.raises(ValueError, match="requires at least 2 modes"):
            dm_single.wormhole_1mode((), output_cutoff=5, leftover_mode=0)
