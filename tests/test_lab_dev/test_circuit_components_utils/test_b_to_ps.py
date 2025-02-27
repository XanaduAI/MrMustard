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

"""Tests for BtoPS."""

# pylint: disable=fixme, missing-function-docstring, pointless-statement

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev import DM, BtoPS, Identity, Ket
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_rho
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from mrmustard.physics.triples import displacement_map_s_parametrized_Abc


class TestBtoPS:
    r"""
    Tests for the ``BtoPS`` class.
    """

    modes = [[0], [1, 2], [7, 9]]
    s = [0, -1, 1]

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        dsmap = BtoPS(modes, s)
        assert dsmap.name == "BtoPS"
        assert dsmap.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_adjoint(self):
        btops = BtoPS([0], 0)
        adjoint_btops = btops.adjoint

        bras = btops.wires.bra.indices
        kets = btops.wires.ket.indices
        assert adjoint_btops.ansatz == btops.ansatz.reorder(kets + bras).conj
        assert adjoint_btops.wires == btops.wires.adjoint
        assert adjoint_btops.parameters.s == btops.parameters.s

    def test_dual(self):
        btops = BtoPS([0], 0)
        dual_btops = btops.dual

        ok = btops.wires.ket.output.indices
        ik = btops.wires.ket.input.indices
        ib = btops.wires.bra.input.indices
        ob = btops.wires.bra.output.indices
        assert dual_btops.ansatz == btops.ansatz.reorder(ib + ob + ik + ok).conj
        assert dual_btops.wires == btops.wires.dual
        assert dual_btops.parameters.s == btops.parameters.s

    def test_inverse(self):
        btops = BtoPS([0], 0)
        inv_btops = btops.inverse()
        assert (btops >> inv_btops).ansatz == Identity([0]).contract(Identity([0]).adjoint).ansatz

    def test_representation(self):
        ansatz = BtoPS(modes=[0], s=0).ansatz
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        assert math.allclose(ansatz.A[0], A_correct)
        assert math.allclose(ansatz.b[0], b_correct)
        assert math.allclose(ansatz.c[0], c_correct)

        ansatz2 = BtoPS(modes=[5, 10], s=1).ansatz
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=1, n_modes=2)
        assert math.allclose(ansatz2.A[0], A_correct)
        assert math.allclose(ansatz2.b[0], b_correct)
        assert math.allclose(ansatz2.c[0], c_correct)

    def testBtoPS_contraction_with_state(self):
        # The init state cov and means comes from the random state 'state = Gaussian(1) >> Dgate([0.2], [0.3])'
        state_cov = np.array([[0.32210229, -0.99732956], [-0.99732956, 6.1926484]])
        state_means = np.array([0.4, 0.6])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=[0], triple=(A, b, c))
        state_bargmann_triple = state.bargmann_triple()

        # get new triple by right shift
        state_after = state >> BtoPS(modes=[0], s=0)
        A1, b1, c1 = state_after.bargmann_triple(batched=True)

        # get new triple by contraction
        Ds_bargmann_triple = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        A2, b2, c2 = complex_gaussian_integral_2(
            state_bargmann_triple, Ds_bargmann_triple, idx1=[0, 1], idx2=[1, 3]
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
            ]
        )
        state_means = np.array([0.28284271, 0.0, 0.42426407, 0.0])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=[0, 1], triple=(A, b, c))
        state_bargmann_triple = state.bargmann_triple()

        # get new triple by right shift
        state_after = state >> BtoPS(modes=[0, 1], s=0)
        A1, b1, c1 = state_after.bargmann_triple(batched=True)

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
        assert math.allclose((psi >> BtoPS([0], 1)).ansatz(0, 0), [1.0])
