# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``Kgate`` class."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab.states import DM, Coherent, Ket, Number
from mrmustard.lab.transformations import Kgate, Rgate
from mrmustard.parameters import Variable


class TestKgate:
    r"""
    Tests for the ``Kgate`` class.
    """

    @staticmethod
    def _g(n, normal_ordered=True):
        return n * (n - 1) if normal_ordered else n * n

    def test_init(self):
        gate = Kgate(0, 0.3)
        assert gate.name == "Kgate"
        assert gate.parameters.kappa.value == 0.3
        assert gate.modes == (0,)
        assert gate.normal_ordered is True
        assert Kgate(0, 0.3, normal_ordered=False).normal_ordered is False

        with pytest.raises(AttributeError, match="no ansatz factory"):
            _ = gate.ansatz

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_diagonal_on_number_state(self, normal_ordered):
        "Kerr acts as a pure phase on number eigenstates."
        kappa = 0.4
        for n in range(4):
            psi = Number(0, n) >> Kgate(0, kappa, normal_ordered=normal_ordered)
            assert isinstance(psi, Ket)
            arr = psi.fock_array(5)
            expected = np.zeros(5, dtype=complex)
            expected[n] = np.exp(1j * kappa * self._g(n, normal_ordered))
            assert math.allclose(arr, expected)

    def test_vacuum_and_one_photon_invariant_when_normal_ordered(self):
        "With normal_ordered=True, |0> and |1> pick up no phase (g(0)=g(1)=0)."
        for n in (0, 1):
            psi = Number(0, n) >> Kgate(0, 0.7)
            arr = psi.fock_array(3)
            expected = np.zeros(3, dtype=complex)
            expected[n] = 1.0
            assert math.allclose(arr, expected)

    def test_preserves_norm(self):
        "Kerr is diagonal with unit-modulus entries, so it preserves the Fock norm."
        psi = Coherent(0, 0.7 + 0.2j).to_fock(shape=12)
        out = psi >> Kgate(0, 0.25)
        assert isinstance(out, Ket)
        assert math.allclose(out.probability, psi.probability)

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_superposition_phase(self, normal_ordered):
        "Kerr imparts exp(i kappa g(n)) on each Fock component."
        kappa = 0.3
        psi = Number(0, 0) + Number(0, 2)
        before = psi.fock_array(3)
        after = (psi >> Kgate(0, kappa, normal_ordered=normal_ordered)).fock_array(3)
        phases = np.exp(1j * kappa * self._g(np.arange(3), normal_ordered))
        assert math.allclose(after, before * phases)

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_dm_action(self, normal_ordered):
        "On a DM, Kerr multiplies rho[bra,ket] by exp(i kappa (g(ket) - g(bra)))."
        kappa = 0.2
        rho = (Number(0, 0) + Number(0, 1)).dm()
        before = rho.fock_array(3)
        gate = Kgate(0, kappa, normal_ordered=normal_ordered)
        after = (rho >> gate).fock_array(3)
        assert isinstance(rho >> gate, DM)
        n = np.arange(3)
        gn = self._g(n, normal_ordered)
        factor = np.exp(1j * kappa * (gn[None, :] - gn[:, None]))
        assert math.allclose(after, before * factor)

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_consistent_with_ket_vs_dm(self, normal_ordered):
        "Applying Kerr to a ket then taking the DM matches applying it to the DM."
        kappa = 0.35
        psi = Coherent(0, 0.5)
        gate = Kgate(0, kappa, normal_ordered=normal_ordered)
        via_ket = (psi >> gate).dm().fock_array(8)
        via_dm = (psi.dm() >> gate).fock_array(8)
        assert math.allclose(via_ket, via_dm)

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_zero_kappa_is_identity(self, normal_ordered):
        psi = Coherent(0, 0.6).to_fock(shape=10)
        before = psi.fock_array()
        after = (psi >> Kgate(0, 0.0, normal_ordered=normal_ordered)).fock_array()
        assert math.allclose(after, before)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_state_batch(self, batch_shape):
        "State batch dims flow through unchanged; Kerr preserves the Fock norm."
        alpha = math.broadcast_to(0.5 + 0.5j, batch_shape)
        psi_in = Coherent(0, alpha).to_fock(shape=12)
        psi_out = psi_in >> Kgate(0, 0.2)
        assert isinstance(psi_out, Ket)
        assert math.allclose(psi_out.probability, psi_in.probability)

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_kappa_batch(self, normal_ordered):
        "Batched kappa produces leading batch axis on the output."
        kappas = math.astensor([0.0, 0.3, 0.7])
        psi = Number(0, 2)
        out = psi >> Kgate(0, kappas, normal_ordered=normal_ordered)
        arr = out.fock_array(3)
        expected = np.zeros((3, 3), dtype=complex)
        for i, k in enumerate([0.0, 0.3, 0.7]):
            expected[i, 2] = np.exp(1j * k * self._g(2, normal_ordered))
        assert math.allclose(arr, expected)

    def test_trainable_parameters(self):
        gate1 = Kgate(0, 0.1)
        kappa_var = Variable(0.1, "kappa", dtype=math.float64)
        gate2 = Kgate(0, kappa=kappa_var)

        with pytest.raises(AttributeError):
            gate1.parameters.kappa.value = 0.2

        gate2.parameters.kappa.value = 0.5
        assert gate2.parameters.kappa.value == 0.5

    @pytest.mark.parametrize("normal_ordered", [True, False])
    def test_compose_kerr_gates(self, normal_ordered):
        "Two Kerr gates on the same mode compose additively in kappa."
        psi = Number(0, 0) + Number(0, 1) + Number(0, 2)
        k_a, k_b = 0.2, 0.3
        out1 = (
            psi
            >> Kgate(0, k_a, normal_ordered=normal_ordered)
            >> Kgate(0, k_b, normal_ordered=normal_ordered)
        )
        out2 = psi >> Kgate(0, k_a + k_b, normal_ordered=normal_ordered)
        assert math.allclose(out1.fock_array(4), out2.fock_array(4))

    def test_n2_vs_nnm1_differ_by_rgate(self):
        "Kgate with n^2 = Kgate with n(n-1) followed by an Rgate(kappa) on |n>, n>=1."
        kappa = 0.3
        psi = Number(0, 0) + Number(0, 1) + Number(0, 2) + Number(0, 3)
        via_n2 = (psi >> Kgate(0, kappa, normal_ordered=False)).fock_array(4)
        via_nnm1 = (psi >> Kgate(0, kappa) >> Rgate(0, kappa)).fock_array(4)
        assert math.allclose(via_n2, via_nnm1)
