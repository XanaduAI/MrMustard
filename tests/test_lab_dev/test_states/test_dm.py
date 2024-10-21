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

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

from itertools import product
import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.circuit_components_utils import TraceOut
from mrmustard.physics.gaussian import vacuum_cov
from mrmustard.lab_dev.states import Coherent, DM, Ket, Number, Vacuum
from mrmustard.lab_dev.transformations import Attenuator, Dgate
from mrmustard.lab_dev.wires import Wires


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


class TestDM:  # pylint:disable=too-many-public-methods
    r"""
    Tests for the ``DM`` class.
    """

    modes = [[[0]], [[0]]]
    x = [[1], [1, -1]]
    y = [[1], [1, -1]]
    coeff = [0.5, 0.3]

    @pytest.mark.parametrize("name", [None, "my_dm"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        state = DM(modes, None, name)

        assert state.name in ("DM0", "DM01", "DM2319") if not name else name
        assert list(state.modes) == sorted(modes)
        assert state.wires == Wires(modes_out_bra=modes, modes_out_ket=modes)

    def test_manual_shape(self):
        dm = Coherent([0, 1], x=[1, 2]).dm()
        assert dm.manual_shape == [None, None, None, None]
        dm.manual_shape[0] = 19
        assert dm.manual_shape == [19, None, None, None]

    def test_auto_shape(self):
        dm = Coherent([0, 1], x=[1, 2]).dm()
        assert dm.auto_shape() == (8, 15, 8, 15)
        dm.manual_shape[0] = 1
        assert dm.auto_shape() == (1, 15, 8, 15)

        dm = Coherent([0, 1], x=1).dm() >> Number([1], 10).dual
        assert dm.auto_shape() == (settings.AUTOSHAPE_MAX, settings.AUTOSHAPE_MAX)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_bargmann(self, modes):
        state_in = Coherent(modes, 1, 2) >> Attenuator([modes[0]], 0.7)
        triple_in = state_in.bargmann_triple()

        state_out = DM.from_bargmann(modes, triple_in, "my_dm")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = Coherent([0, 1], 1).dm()
        with pytest.raises(ValueError):
            DM.from_bargmann(
                [0],
                state01.bargmann_triple(),
                "my_dm",
            )

    def test_from_fock_error(self):
        state01 = Coherent([0, 1], 1).dm()
        state01 = state01.to_fock(2)
        with pytest.raises(ValueError):
            DM.from_fock([0], state01.fock(5), "my_dm", True)

    def test_bargmann_triple_error(self):
        fock = Number([0], n=10).dm()
        with pytest.raises(AttributeError):
            fock.bargmann_triple()

    @pytest.mark.parametrize("modes,x,y,coeff", zip(modes, x, y, coeff))
    def test_normalize(self, modes, x, y, coeff):
        state = Coherent(modes[0], x[0], y[0]).dm()
        for i in range(1, len(modes)):
            state += Coherent(modes[i], x[i], y[i]).dm()
        state *= coeff
        # Bargmann
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)
        # Fock
        state = state.to_fock(5)  # truncated
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_fock(self, modes):
        state_in = Coherent(modes, x=1, y=2) >> Attenuator([modes[0]], 0.8)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock(5, batched=True)

        assert math.allclose(array_in, state_in_fock.representation.array)

        state_out = DM.from_fock(modes, array_in, "my_dm", True)
        assert state_in_fock == state_out

    def test_to_from_phase_space(self):
        state0 = Coherent([0], x=1, y=2) >> Attenuator([0], 1.0)
        cov, means, coeff = state0.phase_space(s=0)  # batch = 1
        assert coeff == 1.0
        assert math.allclose(cov[0], np.eye(2) * settings.HBAR / 2)
        assert math.allclose(means[0], np.array([1.0, 2.0]) * np.sqrt(settings.HBAR * 2))

        # test error
        with pytest.raises(ValueError):
            DM.from_phase_space([0, 1], (cov, means, 1.0))

        cov = vacuum_cov(1)
        means = np.array([1, 2]) * np.sqrt(settings.HBAR * 2 * 0.8)
        state1 = DM.from_phase_space([0], (cov, means, 1.0))
        assert state1 == Coherent([0], 1, 2) >> Attenuator([0], 0.8)

    def test_to_from_quadrature(self):
        modes = [0]
        A0 = np.array([[0, 0], [0, 0]])
        b0 = np.array([0.1 - 0.2j, 0.1 + 0.2j])
        c0 = 0.951229424500714  # z, z^*

        state0 = DM.from_bargmann(modes, (A0, b0, c0))
        Atest, btest, ctest = state0.quadrature_triple()
        state1 = DM.from_quadrature(modes, (Atest[0], btest[0], ctest[0]))
        Atest2, btest2, ctest2 = state1.bargmann_triple()
        assert np.allclose(Atest2, A0)
        assert np.allclose(btest2, b0)
        assert np.allclose(ctest2, c0)

    def test_L2_norms(self):
        state = Coherent([0], x=1).dm() + Coherent([0], x=-1).dm()  # incoherent
        assert len(state._L2_norms) == 2

    def test_L2_norm(self):
        state = Coherent([0], x=1).dm()
        assert state.L2_norm == 1

    def test_probability(self):
        state1 = Coherent([0], x=1).dm()
        assert state1.probability == 1
        assert state1.to_fock(20).probability == 1

        state2 = Coherent([0], x=1).dm() / 3 + 2 * Coherent([0], x=-1).dm() / 3
        assert state2.probability == 1
        assert math.allclose(state2.to_fock(20).probability, 1)

        state3 = Number([0], n=1, cutoffs=2).dm() / 2 + Number([0], n=2).dm() / 2
        assert math.allclose(state3.probability, 1)

    def test_probability_from_ket(self):
        ket_state = Vacuum([0, 1]) >> Number([0], n=1).dual
        dm_state = ket_state.dm()
        assert dm_state.probability == ket_state.probability

    def test_purity(self):
        state = Coherent([0], 1, 2).dm()
        assert math.allclose(state.purity, 1)
        assert state.is_pure

    def test_quadrature_single_mode_dm(self):
        x, y = 1, 2
        state = Coherent(modes=[0], x=x, y=y).dm()
        q = np.linspace(-10, 10, 100)
        quad = math.transpose(math.astensor([q, q + 1]))
        ket = coherent_state_quad(q + 1, x, y)
        bra = np.conj(coherent_state_quad(q, x, y))
        assert math.allclose(state.quadrature(quad), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(quad), bra * ket)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra) ** 2)

    def test_quadrature_multimode_dm(self):
        x, y = 1, 2
        state = Coherent(modes=[0, 1], x=x, y=y).dm()
        q = np.linspace(-10, 10, 100)
        quad = math.tile(math.astensor(list(product(q, repeat=2))), (1, 2))
        ket = math.kron(coherent_state_quad(q, x, y), coherent_state_quad(q, x, y))
        bra = math.kron(
            np.conj(coherent_state_quad(q, x, y)), np.conj(coherent_state_quad(q, x, y))
        )
        assert math.allclose(state.quadrature(quad), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)

        quad_slice = math.transpose(math.astensor([q, q, q + 1, q + 1]))
        q_slice = math.transpose(math.astensor([q] * state.n_modes))
        ket_slice = coherent_state_quad(q + 1, x, y) * coherent_state_quad(q + 1, x, y)
        bra_slice = np.conj(coherent_state_quad(q, x, y)) * np.conj(coherent_state_quad(q, x, y))

        assert math.allclose(state.to_fock(40).quadrature(quad_slice), bra_slice * ket_slice)
        assert math.allclose(
            state.to_fock(40).quadrature_distribution(q_slice), math.abs(bra_slice) ** 2
        )

    def test_quadrature_multivariable_dm(self):
        x, y = 1, 2
        state = Coherent(modes=[0, 1], x=x, y=y).dm()
        q1 = np.linspace(-10, 10, 100)
        q2 = np.linspace(-10, 10, 100)
        quad = np.array([[qa, qb] for qa in q1 for qb in q2])
        psi_q = math.outer(coherent_state_quad(q1, x, y), coherent_state_quad(q2, x, y))
        assert math.allclose(state.quadrature_distribution(quad).reshape(100, 100), abs(psi_q) ** 2)

    def test_quadrature_batch(self):
        x1, y1, x2, y2 = 1, 2, -1, -2
        state = (Coherent(modes=[0], x=x1, y=y1) + Coherent(modes=[0], x=x2, y=y2)).dm()
        q = np.linspace(-10, 10, 100)
        quad = math.transpose(math.astensor([q, q + 1]))
        ket = coherent_state_quad(q + 1, x1, y1) + coherent_state_quad(q + 1, x2, y2)
        bra = np.conj(coherent_state_quad(q, x1, y1) + coherent_state_quad(q, x2, y2))
        assert math.allclose(state.quadrature(quad), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(quad), bra * ket)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra) ** 2)

    def test_expectation_bargmann_ket(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])
        dm = ket.dm()

        k0 = Coherent([0], x=1, y=2)
        k1 = Coherent([1], x=1, y=3)
        k01 = Coherent([0, 1], x=1, y=[2, 3])

        res_k0 = (dm @ k0.dual @ k0.dual.adjoint) >> TraceOut([1])
        res_k1 = (dm @ k1.dual @ k1.dual.adjoint) >> TraceOut([0])
        res_k01 = dm @ k01.dual @ k01.dual.adjoint

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01.representation.c[0])

    def test_expectation_bargmann_dm(self):
        dm0 = Coherent([0], x=1, y=2).dm()
        dm1 = Coherent([1], x=1, y=3).dm()
        dm01 = Coherent([0, 1], x=1, y=[2, 3]).dm()

        res_dm0 = (dm01 @ dm0.dual) >> TraceOut([1])
        res_dm1 = (dm01 @ dm1.dual) >> TraceOut([0])
        res_dm01 = dm01 >> dm01.dual

        assert math.allclose(dm01.expectation(dm0), res_dm0)
        assert math.allclose(dm01.expectation(dm1), res_dm1)
        assert math.allclose(dm01.expectation(dm01), res_dm01)

    def test_expectation_bargmann_u(self):
        dm = Coherent([0, 1], x=1, y=[2, 3]).dm()
        u0 = Dgate([0], x=0.1)
        u1 = Dgate([1], x=0.2)
        u01 = Dgate([0, 1], x=[0.3, 0.4])

        res_u0 = (dm @ u0) >> TraceOut([0, 1])
        res_u1 = (dm @ u1) >> TraceOut([0, 1])
        res_u01 = (dm @ u01) >> TraceOut([0, 1])

        assert math.allclose(dm.expectation(u0), res_u0)
        assert math.allclose(dm.expectation(u1), res_u1)
        assert math.allclose(dm.expectation(u01), res_u01)

    def test_expectation_fock(self):
        ket = Coherent([0, 1], x=1, y=[2, 3]).to_fock(10)
        dm = ket.dm()

        k0 = Coherent([0], x=1, y=2).to_fock(10)
        k1 = Coherent([1], x=1, y=3).to_fock(10)
        k01 = Coherent([0, 1], x=1, y=[2, 3]).to_fock(10)

        res_k0 = (dm @ k0.dual @ k0.dual.adjoint) >> TraceOut([1])
        res_k1 = (dm @ k1.dual @ k1.dual.adjoint) >> TraceOut([0])
        res_k01 = dm @ k01.dual >> k01.dual.adjoint

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01)

        dm0 = Coherent([0], x=1, y=2).to_fock(10).dm()
        dm1 = Coherent([1], x=1, y=3).to_fock(10).dm()
        dm01 = Coherent([0, 1], x=1, y=[2, 3]).to_fock(10).dm()

        res_dm0 = (dm @ dm0.dual) >> TraceOut([1])
        res_dm1 = (dm @ dm1.dual) >> TraceOut([0])
        res_dm01 = dm >> dm01.dual

        assert math.allclose(dm.expectation(dm0), res_dm0)
        assert math.allclose(dm.expectation(dm1), res_dm1)
        assert math.allclose(dm.expectation(dm01), res_dm01)

        u0 = Dgate([0], x=0.1).to_fock(10)
        u1 = Dgate([1], x=0.2).to_fock(10)
        u01 = Dgate([0, 1], x=[0.3, 0.4]).to_fock(10)

        res_u0 = (dm @ u0) >> TraceOut([0, 1])
        res_u1 = (dm @ u1) >> TraceOut([0, 1])
        res_u01 = (dm @ u01) >> TraceOut([0, 1])

        assert math.allclose(dm.expectation(u0), res_u0)
        assert math.allclose(dm.expectation(u1), res_u1)
        assert math.allclose(dm.expectation(u01), res_u01)

    def test_expectation_error(self):
        dm = Coherent([0, 1], x=1, y=[2, 3]).dm()

        op1 = Attenuator([0])
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            dm.expectation(op1)

        op2 = CircuitComponent(wires=[(), (), (1,), (0,)])
        with pytest.raises(ValueError, match="different modes"):
            dm.expectation(op2)

        op3 = Dgate([2])
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            dm.expectation(op3)

    def test_rshift(self):
        ket = Coherent([0, 1], 1)
        unitary = Dgate([0], 1)
        u_component = CircuitComponent._from_attributes(
            unitary.representation, unitary.wires, unitary.name
        )  # pylint: disable=protected-access
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent._from_attributes(
            channel.representation, channel.wires, channel.name
        )  # pylint: disable=protected-access

        dm = ket >> channel

        # gates
        assert isinstance(dm, DM)
        assert isinstance(dm >> unitary >> channel, DM)
        assert isinstance(dm >> channel >> unitary, DM)
        assert isinstance(dm >> u_component, CircuitComponent)
        assert isinstance(dm >> ch_component, CircuitComponent)

        # measurements
        assert isinstance(dm >> Coherent([0], 1).dual, DM)
        assert isinstance(dm >> Coherent([0], 1).dm().dual, DM)

    @pytest.mark.parametrize("modes", [[5], [1, 2]])
    def test_random(self, modes):
        m = len(modes)
        dm = DM.random(modes)
        A = dm.representation.A[0]
        Gamma = A[:m, m:]
        Lambda = A[m:, m:]
        Temp = Gamma + math.conj(Lambda.T) @ math.inv(1 - Gamma.T) @ Lambda
        assert np.all(
            np.linalg.eigvals(Gamma) >= 0
        )  # checks if the off-diagonal block of dm is PSD
        assert np.all(np.linalg.eigvals(Gamma) < 1)
        assert np.all(np.linalg.eigvals(Temp) < 1)

    @pytest.mark.parametrize("modes", [[9, 2], [0, 1, 2, 3, 4]])
    def test_is_positive(self, modes):
        assert (Ket.random(modes) >> Attenuator(modes)).is_positive
        A = np.zeros([2 * len(modes), 2 * len(modes)])
        A[0, -1] = 1.0
        rho = DM.from_bargmann(
            modes, [A, [complex(0)] * 2 * len(modes), [complex(1)]]
        )  # this test fails at the hermitian check
        assert not rho.is_positive

    @pytest.mark.parametrize("modes", [range(10), [0, 1]])
    def test_is_physical(self, modes):
        rho = DM.random(modes)
        assert rho.is_physical
        rho = 2 * rho
        assert not rho.is_physical
        assert Ket.random(modes).dm().is_physical
