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

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement
import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab_dev import (
    Attenuator,
    CircuitComponent,
    Coherent,
    Dgate,
    DM,
    Ket,
    Number,
    TraceOut,
    Vacuum,
)
from mrmustard.physics.gaussian import vacuum_cov
from mrmustard.physics.representations import Representation
from mrmustard.physics.triples import coherent_state_Abc
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


class TestDM:  # pylint:disable=too-many-public-methods
    r"""
    Tests for the ``DM`` class.
    """

    @pytest.mark.parametrize("name", [None, "my_dm"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {2, 3, 19}])
    def test_init(self, name, modes):
        state = DM.from_ansatz(modes, None, name)

        assert state.name in ("DM0", "DM01", "DM2319") if not name else name
        assert list(state.modes) == sorted(modes)
        assert state.wires == Wires(modes_out_bra=modes, modes_out_ket=modes)

    def test_manual_shape(self):
        dm = Coherent(0, x=1).dm()
        assert dm.manual_shape == [None, None]
        dm.manual_shape[0] = 19
        assert dm.manual_shape == [19, None]

    def test_auto_shape(self):
        dm = Coherent(0, x=1).dm()
        assert dm.auto_shape() == (8, 8)
        dm.manual_shape[0] = 1
        assert dm.auto_shape() == (1, 8)

        dm = Coherent(0, x=1).dm() >> Number(1, 10).dual
        assert dm.auto_shape() == (8, 11, 8, 11)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    def test_to_from_bargmann(self, modes):
        state_in = Coherent(modes, 1, 2) >> Attenuator(modes, 0.7)
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
    def test_normalize(self, coeff):
        state = Coherent(0, 1, 1).dm() + Coherent(0, -1, -1).dm()
        state *= coeff
        # Bargmann
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)
        # Fock
        state = state.to_fock(5)  # truncated
        normalized = state.normalize()
        assert math.allclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("modes", [0, 1, 7])
    def test_to_from_fock(self, modes):
        state_in = Coherent(modes, x=1, y=2) >> Attenuator(modes, 0.8)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock_array(5)

        assert math.allclose(array_in, state_in_fock.ansatz.array)

        state_out = DM.from_fock((modes,), array_in, "my_dm")
        assert state_in_fock == state_out

    def test_to_from_phase_space(self):
        state0 = Coherent(0, x=1, y=2) >> Attenuator(0, 1.0)
        cov, means, coeff = state0.phase_space(s=0)  # batch = 1
        assert math.allclose(coeff, math.atleast_1d(1.0))
        assert math.allclose(cov, math.eye(2) * settings.HBAR / 2)
        assert math.allclose(means, math.astensor([1.0, 2.0]) * math.sqrt(settings.HBAR * 2))

        # test error
        with pytest.raises(ValueError):
            DM.from_phase_space((0, 1), (cov, means, 1.0))

        cov = vacuum_cov(1)
        means = np.array([1, 2]) * np.sqrt(settings.HBAR * 2 * 0.8)
        state1 = DM.from_phase_space([0], (cov, means, 1.0))
        assert state1 == Coherent(0, 1, 2) >> Attenuator(0, 0.8)

    def test_to_from_quadrature(self):
        modes = (0,)
        A0 = np.array([[0, 0], [0, 0]])
        b0 = np.array([0.1 - 0.2j, 0.1 + 0.2j])
        c0 = math.astensor(0.951229424500714)  # z, z^*

        state0 = DM.from_bargmann(modes, (A0, b0, c0))
        Atest, btest, ctest = state0.quadrature_triple()
        state1 = DM.from_quadrature(modes, (Atest, btest, ctest))
        Atest2, btest2, ctest2 = state1.bargmann_triple()
        assert math.allclose(Atest2, A0)
        assert math.allclose(btest2, b0)
        assert math.allclose(ctest2, c0)

    def test_L2_norms(self):
        state = Coherent(0, x=1).dm() + Coherent(0, x=-1).dm()  # incoherent
        assert len(state._L2_norms) == 2

    def test_L2_norm(self):
        state = Coherent(0, x=1).dm()
        assert math.allclose(state.L2_norm, 1)

    def test_probability(self):
        state1 = Coherent(0, x=1).dm()
        assert math.allclose(state1.probability, 1)
        assert math.allclose(state1.to_fock(20).probability, 1)

        state2 = Coherent(0, x=1).dm() / 3 + 2 * Coherent(0, x=-1).dm() / 3
        assert math.allclose(state2.probability, 1)
        assert math.allclose(state2.to_fock(20).probability, 1)

        state3 = Number(0, n=1, cutoff=2).dm() / 2 + Number(0, n=2).dm() / 2
        assert math.allclose(state3.probability, 1)

    def test_probability_from_ket(self):
        ket_state = Vacuum((0, 1)) >> Number(0, n=1).dual
        dm_state = ket_state.dm()
        assert dm_state.probability == ket_state.probability

    def test_purity(self):
        state = Coherent(mode=0, x=1, y=2).dm()
        assert math.allclose(state.purity, 1)
        assert state.is_pure

    def test_quadrature_single_mode_dm(self):
        x, y = 1, 2
        state = Coherent(mode=0, x=x, y=y).dm()
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
        state = (Coherent(mode=0, x=x, y=y) >> Coherent(mode=1, x=x, y=y)).dm()
        q = np.linspace(-10, 10, 100)
        ket = math.kron(coherent_state_quad(q, x, y), coherent_state_quad(q, x, y))
        bra = math.kron(
            np.conj(coherent_state_quad(q, x, y)), np.conj(coherent_state_quad(q, x, y))
        )
        assert math.allclose(state.quadrature(q, q, q, q), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)

        ket_slice = math.kron(coherent_state_quad(q + 1, x, y), coherent_state_quad(q + 1, x, y))
        bra_slice = math.kron(
            np.conj(coherent_state_quad(q, x, y)), np.conj(coherent_state_quad(q, x, y))
        )

        assert math.allclose(
            state.to_fock(40).quadrature(q, q, q + 1, q + 1), bra_slice * ket_slice
        )
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra_slice) ** 2)

    def test_quadrature_multivariable_dm(self):
        x, y = 1, 2
        state = Coherent(mode=0, x=x, y=y).dm() >> Coherent(mode=1, x=x, y=y).dm()
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
        state = Ket.from_bargmann((0,), (A, b, c)).dm()
        q = np.linspace(-10, 10, 100)

        ket = math.astensor([coherent_state_quad(q, x1, y1), coherent_state_quad(q, x2, y2)]).T
        bra = math.astensor(
            [np.conj(coherent_state_quad(q, x1, y1)), np.conj(coherent_state_quad(q, x2, y2))]
        ).T

        assert math.allclose(state.quadrature(q, q), bra * ket)
        assert math.allclose(state.quadrature_distribution(q), math.abs(bra) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(q, q), bra * ket)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), math.abs(bra) ** 2)

    def test_expectation_bargmann_ket(self):
        ket = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)
        dm = ket.dm()

        k0 = Coherent(0, x=1, y=2)
        k1 = Coherent(1, x=1, y=3)
        k01 = Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)

        res_k0 = (dm.contract(k0.dual).contract(k0.dual.adjoint)) >> TraceOut(1)
        res_k1 = (dm.contract(k1.dual).contract(k1.dual.adjoint)) >> TraceOut(0)
        res_k01 = dm.contract(k01.dual).contract(k01.dual.adjoint)

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01.ansatz.c)

    def test_expectation_bargmann_dm(self):
        dm0 = Coherent(0, x=1, y=2).dm()
        dm1 = Coherent(1, x=1, y=3).dm()
        dm01 = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).dm()

        res_dm0 = (dm01.contract(dm0.dual)) >> TraceOut(1)
        res_dm1 = (dm01.contract(dm1.dual)) >> TraceOut(0)
        res_dm01 = dm01 >> dm01.dual

        assert math.allclose(dm01.expectation(dm0), res_dm0)
        assert math.allclose(dm01.expectation(dm1), res_dm1)
        assert math.allclose(dm01.expectation(dm01), res_dm01)

    def test_expectation_bargmann_u(self):
        dm = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).dm()
        u0 = Dgate(0, x=0.1)
        u1 = Dgate(1, x=0.2)
        u01 = Dgate(0, x=0.3) >> Dgate(1, x=0.4)

        res_u0 = (dm.contract(u0)) >> TraceOut(0) >> TraceOut(1)
        res_u1 = (dm.contract(u1)) >> TraceOut(0) >> TraceOut(1)
        res_u01 = (dm.contract(u01)) >> TraceOut(0) >> TraceOut(1)

        assert math.allclose(dm.expectation(u0), res_u0)
        assert math.allclose(dm.expectation(u1), res_u1)
        assert math.allclose(dm.expectation(u01), res_u01)

    def test_expectation_fock(self):
        ket = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).to_fock(10)
        dm = ket.dm()

        k0 = Coherent(0, x=1, y=2).to_fock(10)
        k1 = Coherent(1, x=1, y=3).to_fock(10)
        k01 = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).to_fock(10)

        res_k0 = (dm.contract(k0.dual).contract(k0.dual.adjoint)) >> TraceOut(1)
        res_k1 = (dm.contract(k1.dual).contract(k1.dual.adjoint)) >> TraceOut(0)
        res_k01 = dm.contract(k01.dual) >> k01.dual.adjoint

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01)

        dm0 = Coherent(0, x=1, y=2).to_fock(10).dm()
        dm1 = Coherent(1, x=1, y=3).to_fock(10).dm()
        dm01 = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).to_fock(10).dm()

        res_dm0 = (dm.contract(dm0.dual)) >> TraceOut(1)
        res_dm1 = (dm.contract(dm1.dual)) >> TraceOut(0)
        res_dm01 = dm >> dm01.dual

        assert math.allclose(dm.expectation(dm0), res_dm0)
        assert math.allclose(dm.expectation(dm1), res_dm1)
        assert math.allclose(dm.expectation(dm01), res_dm01)

        u0 = Dgate(0, x=0.1).to_fock(10)
        u1 = Dgate(1, x=0.2).to_fock(10)
        u01 = (Dgate(0, x=0.3) >> Dgate(1, x=0.4)).to_fock(10)

        res_u0 = (dm.contract(u0)) >> TraceOut(0) >> TraceOut(1)
        res_u1 = (dm.contract(u1)) >> TraceOut(0) >> TraceOut(1)
        res_u01 = (dm.contract(u01)) >> TraceOut(0) >> TraceOut(1)

        assert math.allclose(dm.expectation(u0), res_u0)
        assert math.allclose(dm.expectation(u1), res_u1)
        assert math.allclose(dm.expectation(u01), res_u01)

    def test_expectation_error(self):
        dm = (Coherent(0, x=1, y=2) >> Coherent(1, x=1, y=3)).dm()

        op1 = Attenuator(0)
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            dm.expectation(op1)

        op2 = CircuitComponent(Representation(wires=Wires(set(), set(), {1}, {0})))
        with pytest.raises(ValueError, match="different modes"):
            dm.expectation(op2)

        op3 = Dgate(2)
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            dm.expectation(op3)

    def test_fock_distribution(self):
        state = Coherent(0, x=1, y=2)
        assert math.allclose(state.fock_distribution(10), state.dm().fock_distribution(10))

    def test_rshift(self):
        ket = Coherent(0, 1) >> Coherent(1, 1)
        unitary = Dgate(0, 1)
        u_component = CircuitComponent(unitary.representation, unitary.name)
        channel = Attenuator(1, 1)
        ch_component = CircuitComponent(channel.representation, channel.name)

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
        dm = DM.random(modes)
        A = dm.ansatz.A
        Gamma = A[..., :m, m:]
        Lambda = A[..., m:, m:]
        Temp = Gamma + math.conj(Lambda.T) @ math.inv(1 - Gamma.T) @ Lambda
        assert np.all(
            np.linalg.eigvals(Gamma) >= 0
        )  # checks if the off-diagonal block of dm is PSD
        assert np.all(np.linalg.eigvals(Gamma) < 1)
        assert np.all(np.linalg.eigvals(Temp) < 1)

    def test_is_positive(self):
        assert (Ket.random((2, 9)) >> Attenuator(2) >> Attenuator(9)).is_positive
        A = np.zeros((4, 4))
        A[0, -1] = 1.0
        rho = DM.from_bargmann(
            (2, 9), [A, [complex(0)] * 4, complex(1)]
        )  # this test fails at the hermitian check
        assert not rho.is_positive

    @pytest.mark.parametrize("modes", [tuple(range(10)), (0, 1)])
    def test_is_physical(self, modes):
        rho = DM.random(modes)
        assert rho.is_physical
        rho = 2 * rho
        assert not rho.is_physical
        assert Ket.random(modes).dm().is_physical

    def test_fock_array_ordering(self):  # TODO: linear superpositions
        rho = Number(0, 0) + 1j * Number(0, 1)
        rho = (Number(0, 0) + 1j * Number(0, 1)).dm()
        rho_fock = rho.fock_array(standard_order=True)

        assert math.allclose(
            rho_fock, math.astensor([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]])
        )

    def test_formal_stellar_decomposition(self):
        rho = DM.random([0, 1])
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
        rho = DM.random([0, 1, 2, 3])
        core, phi = rho.formal_stellar_decomposition([0, 3])

        assert rho == core >> phi

    def test_physical_stellar_decomposition(self):
        rho = DM.random([0, 1])
        core, phi = rho.physical_stellar_decomposition([0])

        assert rho == core >> phi
        assert core.is_physical
        assert isinstance(core, Ket)

        A, _, _ = core.ansatz.triple
        assert A[0, 0] == 0

        # 4-mode example
        rho = DM.random([0, 1, 2, 3])
        core, phi = rho.physical_stellar_decomposition([0, 3])

        assert rho == core >> phi
        assert phi.is_physical

    def test_stellar_decomposition_2(self):
        rho = DM.random([0, 1])
        core, phi = rho.stellar_decomposition_2([0])

        assert rho == core >> phi
        assert core.is_physical
        assert isinstance(core, DM)

        A, _, _ = core.ansatz.triple
        assert A[0, 0] == 0

        # 4-mode example
        rho = DM.random([0, 1, 2, 3])
        core, phi = rho.stellar_decomposition_2([0, 3])

        assert rho == core >> phi
        assert phi.is_physical
