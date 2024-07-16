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

"""Tests for the state subpackage."""

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement


import numpy as np
import pytest

from mrmustard import settings

from mrmustard.lab_dev.states import Coherent, QuadratureEigenstate

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF


class TestQuadratureEigenstate:
    r"""
    Tests for the ``QuadratureEigenstate`` class.
    """

    modes = [[1], [0, 1], [1, 5]]
    x = [[1], 1, [2]]
    phi = [3, [4], 1]

    @pytest.mark.parametrize("modes,x,phi", zip(modes, x, phi))
    def test_init1(self, modes, x, phi):
        state = QuadratureEigenstate(modes, x, phi)

        assert state.name == "QuadratureEigenstate"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``x``"):
            QuadratureEigenstate(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``phi``"):
            QuadratureEigenstate(modes=[0, 1], x=1, phi=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = QuadratureEigenstate([0, 1], 1, 1)
        state2 = QuadratureEigenstate([0, 1], 1, 1, x_trainable=True, x_bounds=(0, 2))
        state3 = QuadratureEigenstate([0, 1], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.x.value = 3

        state2.x.value = 2
        assert state2.x.value == 2

        state3.phi.value = 2
        assert state3.phi.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            QuadratureEigenstate(modes=[0], x=[0.1, 0.2]).representation

    def test_with_coherent(self):
        val0 = Coherent([0], 0, 0) >> QuadratureEigenstate([0], 0, 0).dual
        val1 = Coherent([0], 1, 0) >> QuadratureEigenstate([0], 2, 0).dual
        assert np.allclose(val0, val1)

        val0 = Coherent([0], 1, 0) >> QuadratureEigenstate([0], 0, 0).dual
        val1 = Coherent([0], 1, 0) >> QuadratureEigenstate([0], 2, 0).dual
        assert val1 > val0
