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

""" Tests for circuit components utils. """

# pylint: disable=fixme, missing-function-docstring, protected-access, pointless-statement

import pytest
import numpy as np

from mrmustard import math
from mrmustard.lab_dev.circuit_components_utils import _DsMap, _BtoQMap
from mrmustard.lab_dev.states.base import DM
from mrmustard.physics.triples import displacement_map_s_parametrized_Abc
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.physics.gaussian_integrals import (
    contract_two_Abc,
    real_gaussian_integral,
    complex_gaussian_integral,
    join_Abc,
    join_Abc_real,
)


class TestDsMap:
    r"""
    Tests for the ``DsMap`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    s = [0, -1, 1]

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        dsmap = _DsMap(modes, s)  # pylint: disable=protected-access

        assert dsmap.name == "_DsMap"
        assert dsmap.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_representation(self):
        rep1 = _DsMap(modes=[0], s=0).representation  # pylint: disable=protected-access
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        assert math.allclose(rep1.A[0], A_correct)
        assert math.allclose(rep1.b[0], b_correct)
        assert math.allclose(rep1.c[0], c_correct)

        rep2 = _DsMap(modes=[5, 10], s=1).representation  # pylint: disable=protected-access
        A_correct, b_correct, c_correct = displacement_map_s_parametrized_Abc(s=1, n_modes=2)
        assert math.allclose(rep2.A[0], A_correct)
        assert math.allclose(rep2.b[0], b_correct)
        assert math.allclose(rep2.c[0], c_correct)

    def test_dsmap_contraction_with_state(self):
        # The init state cov and means comes from the random state 'state = Gaussian(1) >> Dgate([0.2], [0.3])'
        state_cov = np.array([[0.32210229, -0.99732956], [-0.99732956, 6.1926484]])
        state_means = np.array([0.4, 0.6])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=[0], triple=(A, b, c))
        state_bargmann_triple = (A, b, c)

        # get new triple by right shift
        state_after = state >> _DsMap(modes=[0], s=0)  # pylint: disable=protected-access
        A1, b1, c1 = state_after.bargmann_triple

        # get new triple by contraction
        Ds_bargmann_triple = displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        A2, b2, c2 = contract_two_Abc(
            state_bargmann_triple, Ds_bargmann_triple, idx1=[0, 1], idx2=[1, 3]
        )

        assert math.allclose(A1[0], A2)
        assert math.allclose(b1[0], b2)
        assert math.allclose(c1[0], c2)

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
        state_bargmann_triple = (A, b, c)

        # get new triple by right shift
        state_after = state >> _DsMap(modes=[0, 1], s=0)  # pylint: disable=protected-access
        A1, b1, c1 = state_after.bargmann_triple

        # get new triple by contraction
        Ds_bargmann_triple = displacement_map_s_parametrized_Abc(s=0, n_modes=2)
        A2, b2, c2 = contract_two_Abc(
            state_bargmann_triple, Ds_bargmann_triple, idx1=[0, 1, 2, 3], idx2=[2, 3, 6, 7]
        )

        assert math.allclose(A1[0], A2)
        assert math.allclose(b1[0], b2)
        assert math.allclose(c1[0], c2)

    def test_btoqmap_works_correctly_by_applying_it_twice_on_a_state(self):
        A0 = np.array([[0.5, 0.3], [0.3, 0.5]])
        b0 = np.zeros(2)
        c0 = 1.0

        modes = [0, 1]
        QtoBMap_CC1 = _BtoQMap(modes)
        step1A, step1b, step1c = (
            QtoBMap_CC1.representation.A[0],
            QtoBMap_CC1.representation.b[0],
            QtoBMap_CC1.representation.c[0],
        )
        Ainter, binter, cinter = complex_gaussian_integral(
            join_Abc((A0, b0, c0), (step1A, step1b, step1c)),
            idx_z=[0, 1],
            idx_zconj=[2, 3],
            measure=-1,
        )
        QtoBMap_CC2 = _BtoQMap(modes).dual
        step2A, step2b, step2c = (
            QtoBMap_CC2.representation.A[0],
            QtoBMap_CC2.representation.b[0],
            QtoBMap_CC2.representation.c[0],
        )

        new_A, new_b, new_c = join_Abc_real(
            (Ainter, binter, cinter), (step2A, step2b, step2c), [0, 1], [0, 1]
        )

        Af, bf, cf = real_gaussian_integral((new_A, new_b, new_c), idx=[0, 1])

        assert math.allclose(A0, Af)
        assert math.allclose(b0, bf)
        assert math.allclose(c0, cf)
