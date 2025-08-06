# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BtoChar."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab import DM, BtoChar, Identity, Ket
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_rho
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from mrmustard.physics.triples import displacement_map_s_parametrized_Abc


class TestBtoChar:
    r"""
    Tests for the ``BtoChar`` class.
    """

    modes = [(0,), (1, 2), (7, 9)]
    s = [0, -1, 1]

    def test_adjoint(self):
        btochar = BtoChar(0, 0)
        adjoint_btochar = btochar.adjoint

        bras = btochar.wires.bra.indices
        kets = btochar.wires.ket.indices
        assert adjoint_btochar.ansatz == btochar.ansatz.reorder(kets + bras).conj
        assert adjoint_btochar.wires == btochar.wires.adjoint
        assert adjoint_btochar.parameters.s == btochar.parameters.s

    def test_BtoChar_contraction_with_state(self):
        # The init state cov and means comes from the random state 'state = Gaussian(1) >> Dgate([0.2], [0.3])'
        state_cov = np.array([[0.32210229, -0.99732956], [-0.99732956, 6.1926484]])
        state_means = np.array([0.4, 0.6])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=(0,), triple=(A, b, c))
        state_bargmann_triple = state.bargmann_triple()

        # get new triple by right shift
        state_after = state >> BtoChar(modes=(0,), s=0)
        A1, b1, c1 = state_after.bargmann_triple()

        # get new triple by contraction
        Ds_bargmann_triple = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        A2, b2, c2 = complex_gaussian_integral_2(
            state_bargmann_triple,
            Ds_bargmann_triple,
            idx1=[0, 1],
            idx2=[1, 3],
        )

        assert math.allclose(A1, A2)
        assert math.allclose(b1, b2)
        assert math.allclose(c1, c2)

        # The init state cov and means comes from the random state 'state = Gaussian(2) >> Dgate([0.2], [0.3])'
        state_cov = np.array(
            [
                [0.77969414, 0.10437996, 0.72706741, 0.29121535],
                [0.10437996, 0.22846619, 0.1211067, 0.45983868],
                [0.72706741, 0.1211067, 1.02215481, 0.16216756],
                [0.29121535, 0.45983868, 0.16216756, 2.10006],
            ],
        )
        state_means = np.array([0.28284271, 0.0, 0.42426407, 0.0])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=(0, 1), triple=(A, b, c))
        state_bargmann_triple = state.bargmann_triple()

        # get new triple by right shift
        state_after = state >> BtoChar(modes=(0, 1), s=0)
        A1, b1, c1 = state_after.bargmann_triple()

        # get new triple by contraction
        Ds_bargmann_triple = displacement_map_s_parametrized_Abc(s=0, n_modes=2)
        A2, b2, c2 = complex_gaussian_integral_2(
            state_bargmann_triple,
            Ds_bargmann_triple,
            idx1=[0, 1, 2, 3],
            idx2=[2, 3, 6, 7],
        )

        assert math.allclose(A1, A2)
        assert math.allclose(b1, b2)
        assert math.allclose(c1, c2)

        psi = Ket.random([0])
        assert math.allclose((psi >> BtoChar(0, 1)).ansatz(0, 0), 1.0)

    def test_dual(self):
        btochar = BtoChar(0, 0)
        dual_btochar = btochar.dual

        ok = btochar.wires.ket.output.indices
        ik = btochar.wires.ket.input.indices
        ib = btochar.wires.bra.input.indices
        ob = btochar.wires.bra.output.indices
        assert dual_btochar.ansatz == btochar.ansatz.reorder(ib + ob + ik + ok).conj
        assert dual_btochar.wires == btochar.wires.dual
        assert dual_btochar.parameters.s == btochar.parameters.s

    def test_fock_array(self):
        btochar = BtoChar(0, 0)
        with pytest.raises(NotImplementedError):
            btochar.fock_array()

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        dsmap = BtoChar(modes, s)
        assert dsmap.name == "BtoChar"
        assert dsmap.modes == modes

    def test_inverse(self):
        btochar = BtoChar(0, 0)
        inv_btochar = btochar.inverse()
        assert (btochar >> inv_btochar).ansatz == (Identity(0).contract(Identity(0).adjoint)).ansatz

    def test_representation(self):
        ansatz = BtoChar(modes=0, s=0).ansatz
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        assert math.allclose(ansatz.A, A_correct)
        assert math.allclose(ansatz.b, b_correct)
        assert math.allclose(ansatz.c, c_correct)

        ansatz2 = BtoChar(modes=(5, 10), s=1).ansatz
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=1, n_modes=2)
        assert math.allclose(ansatz2.A, A_correct)
        assert math.allclose(ansatz2.b, b_correct)
        assert math.allclose(ansatz2.c, c_correct)
