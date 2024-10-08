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

"""Tests for the ``QuadratureEigenstate`` class."""

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import numpy as np
import pytest

from mrmustard.lab_dev.states import QuadratureEigenstate, Coherent
from mrmustard import settings


class TestQuadratureEigenstate:
    r"""
    Tests for the ``QuadratureEigenstate`` class.
    """

    modes = [[1], [0, 1], [1, 5]]
    x = [[1], 1, [2]]
    phi = [3, [4], 1]
    hbar = [3.0, 4.0]

    def test_auto_shape(self):
        state = QuadratureEigenstate([0], x=1, phi=0)
        assert state.auto_shape() == state.manual_shape

    def test_dual(self):
        state = QuadratureEigenstate([0], x=0, phi=0)
        assert state >> state.dual == np.inf

    @pytest.mark.parametrize("modes,x,phi", zip(modes, x, phi))
    def test_init(self, modes, x, phi):
        state = QuadratureEigenstate(modes, x, phi)

        assert state.name == "QuadratureEigenstate"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)
        assert state.L2_norm == np.inf
        assert np.allclose(state.x.value, x)
        assert np.allclose(state.phi.value, phi)

    def test_init_error(self):
        with pytest.raises(ValueError, match="x"):
            QuadratureEigenstate(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="phi"):
            QuadratureEigenstate(modes=[0, 1], x=1, phi=[2, 3, 4])

    @pytest.mark.parametrize("hbar", hbar)
    def test_probability_hbar(self, hbar):
        with settings(HBAR=2.0):
            q0 = QuadratureEigenstate([0], x=0, phi=0).bargmann_triple()

        with settings(HBAR=hbar):
            q1 = QuadratureEigenstate([0], x=0, phi=0).bargmann_triple()

        assert all(np.allclose(a, b) for a, b in zip(q0, q1))

    def test_representation_error(self):
        with pytest.raises(ValueError):
            QuadratureEigenstate(modes=[0], x=[0.1, 0.2]).ansatz

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

    def test_with_coherent(self):
        val0 = Coherent([0], 0, 0) >> QuadratureEigenstate([0], 0, 0).dual
        val1 = Coherent([0], 1, 0) >> QuadratureEigenstate([0], np.sqrt(2 * settings.HBAR), 0).dual
        assert np.allclose(val0, val1)
