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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.physics.triples import identity_Abc, displacement_map_s_parametrized_Abc
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.physics.gaussian_integrals import contract_two_Abc
from mrmustard.physics.representations import Bargmann
from mrmustard.lab_dev.circuit_components_utils import TraceOut, _DsMap
from mrmustard.lab_dev.states import Coherent, DM
from mrmustard.lab_dev.wires import Wires


# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF


class TestTraceOut:
    r"""
    Tests ``TraceOut`` objects.
    """

    @pytest.mark.parametrize("modes", [[0], [1, 2], [3, 4, 5]])
    def test_init(self, modes):
        tr = TraceOut(modes)

        assert tr.name == "Tr"
        assert tr.wires == Wires(modes_in_bra=set(modes), modes_in_ket=set(modes))
        assert tr.representation == Bargmann(*identity_Abc(len(modes)))

    def test_trace_out_bargmann_states(self):
        state = Coherent([0, 1, 2], x=1)

        assert state >> TraceOut([0]) == Coherent([1, 2], x=1).dm()
        assert state >> TraceOut([1, 2]) == Coherent([0], x=1).dm()

        no_state = state >> TraceOut([0, 1, 2])
        assert no_state.modes == []
        assert no_state.wires == Wires()
        assert np.allclose(no_state.representation.A, [])
        assert np.allclose(no_state.representation.b, [])
        assert np.allclose(no_state.representation.c, 1)

    def test_trace_out_fock_states(self):
        settings.AUTOCUTOFF_MAX_CUTOFF = 10

        state = Coherent([0, 1, 2], x=1).to_fock_component()
        assert state >> TraceOut([0]) == Coherent([1, 2], x=1).to_fock_component().dm()
        assert state >> TraceOut([1, 2]) == Coherent([0], x=1).to_fock_component().dm()

        no_state = state >> TraceOut([0, 1, 2])
        assert no_state.modes == []
        assert no_state.wires == Wires()
        assert np.allclose(no_state.representation.array, [])

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0


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
