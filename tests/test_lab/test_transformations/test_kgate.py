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
from mrmustard.lab.transformations import Kgate
from mrmustard.parameters import Variable


class TestKgate:
    r"""
    Tests for the ``Kgate`` class.
    """

    def test_init(self):
        gate = Kgate(0, 0.3)
        assert gate.name == "Kgate"
        assert gate.parameters.kappa.value == 0.3
        assert gate.modes == (0,)
        assert gate.ansatz is None

    def test_diagonal_on_number_state(self):
        "Kerr acts as a pure phase on number eigenstates."
        kappa = 0.4
        for n in range(4):
            psi = Number(0, n) >> Kgate(0, kappa)
            assert isinstance(psi, Ket)
            arr = psi.fock_array(5)
            expected = np.zeros(5, dtype=complex)
            expected[n] = np.exp(1j * kappa * n**2)
            assert math.allclose(arr, expected)

    def test_preserves_norm(self):
        "Kerr is diagonal with unit-modulus entries, so it preserves the Fock norm."
        psi = Coherent(0, 0.7 + 0.2j).to_fock(shape=12)
        out = psi >> Kgate(0, 0.25)
        assert isinstance(out, Ket)
        assert math.allclose(out.probability, psi.probability)

    def test_superposition_phase(self):
        "Kerr imparts exp(i kappa n^2) on each Fock component."
        kappa = 0.3
        psi = Number(0, 0) + Number(0, 2)
        before = psi.fock_array(3)
        after = (psi >> Kgate(0, kappa)).fock_array(3)
        phases = np.exp(1j * kappa * np.arange(3) ** 2)
        assert math.allclose(after, before * phases)

    def test_dm_action(self):
        "On a DM, Kerr multiplies rho[bra,ket] by exp(i kappa (ket^2 - bra^2))."
        kappa = 0.2
        rho = (Number(0, 0) + Number(0, 1)).dm()
        before = rho.fock_array(3)
        after = (rho >> Kgate(0, kappa)).fock_array(3)
        assert isinstance(rho >> Kgate(0, kappa), DM)
        n = np.arange(3)
        factor = np.exp(1j * kappa * (n[None, :] ** 2 - n[:, None] ** 2))
        assert math.allclose(after, before * factor)

    def test_consistent_with_ket_vs_dm(self):
        "Applying Kerr to a ket then taking the DM matches applying it to the DM."
        kappa = 0.35
        psi = Coherent(0, 0.5)
        via_ket = (psi >> Kgate(0, kappa)).dm().fock_array(8)
        via_dm = (psi.dm() >> Kgate(0, kappa)).fock_array(8)
        assert math.allclose(via_ket, via_dm)

    def test_zero_kappa_is_identity(self):
        psi = Coherent(0, 0.6).to_fock(shape=10)
        before = psi.fock_array()
        after = (psi >> Kgate(0, 0.0)).fock_array()
        assert math.allclose(after, before)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_state_batch(self, batch_shape):
        "State batch dims flow through unchanged; Kerr preserves the Fock norm."
        alpha = math.broadcast_to(0.5 + 0.5j, batch_shape)
        psi_in = Coherent(0, alpha).to_fock(shape=12)
        psi_out = psi_in >> Kgate(0, 0.2)
        assert isinstance(psi_out, Ket)
        assert math.allclose(psi_out.probability, psi_in.probability)

    def test_kappa_batch(self):
        "Batched kappa produces leading batch axis on the output."
        kappas = math.astensor([0.0, 0.3, 0.7])
        psi = Number(0, 2)
        out = psi >> Kgate(0, kappas)
        arr = out.fock_array(3)
        expected = np.zeros((3, 3), dtype=complex)
        for i, k in enumerate([0.0, 0.3, 0.7]):
            expected[i, 2] = np.exp(1j * k * 4)
        assert math.allclose(arr, expected)

    def test_trainable_parameters(self):
        gate1 = Kgate(0, 0.1)
        kappa_var = Variable(0.1, "kappa", dtype=math.float64)
        gate2 = Kgate(0, kappa=kappa_var)

        with pytest.raises(AttributeError):
            gate1.parameters.kappa.value = 0.2

        gate2.parameters.kappa.value = 0.5
        assert gate2.parameters.kappa.value == 0.5

    def test_compose_kerr_gates(self):
        "Two Kerr gates on the same mode compose additively in kappa."
        psi = Number(0, 0) + Number(0, 1) + Number(0, 2)
        out1 = psi >> Kgate(0, 0.2) >> Kgate(0, 0.3)
        out2 = psi >> Kgate(0, 0.5)
        assert math.allclose(out1.fock_array(4), out2.fock_array(4))
