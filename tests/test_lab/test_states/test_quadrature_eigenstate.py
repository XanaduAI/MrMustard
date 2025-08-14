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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab.states import Coherent, QuadratureEigenstate
from mrmustard.physics.wires import ReprEnum


class TestQuadratureEigenstate:
    r"""
    Tests for the ``QuadratureEigenstate`` class.
    """

    modes = [0, 1, 7]
    x = [1, 2, 3]
    phi = [3, 4, 5]
    hbar = [3.0, 4.0]

    def test_auto_shape(self):
        state = QuadratureEigenstate(0, x=1, phi=0)
        assert state.auto_shape() == state.manual_shape

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_dual(self, batch_shape):
        x = math.zeros(batch_shape)
        state = QuadratureEigenstate(0, x=x)
        assert math.all(math.real(state >> state.dual) == np.inf)

    @pytest.mark.parametrize("modes,x,phi", zip(modes, x, phi))
    def test_init(self, modes, x, phi):
        state = QuadratureEigenstate(modes, x, phi)

        assert state.name == "QuadratureEigenstate"
        assert state.modes == (modes,)
        assert state.L2_norm == np.inf

    @pytest.mark.parametrize("hbar", hbar)
    def test_probability_hbar(self, hbar):
        with settings(HBAR=2.0):
            A1, b1, c1 = QuadratureEigenstate(0, x=0, phi=0).bargmann_triple()

        with settings(HBAR=hbar):
            A2, b2, c2 = QuadratureEigenstate(0, x=0, phi=0).bargmann_triple()

        assert math.allclose(A1, A2)
        assert math.allclose(b1, b2)
        assert math.allclose(c1, c2)

    def test_parameter_storage(self):
        state = QuadratureEigenstate(0, 2.5, 1.2)

        assert state.name == "QuadratureEigenstate"
        assert state.modes == (0,)
        
        batch_x = [1.0, 2.0, 3.0]
        batch_phi = [0.1, 0.2, 0.3]
        batch_state = QuadratureEigenstate(0, batch_x, batch_phi)

    def test_with_coherent(self):
        val0 = Coherent(0, 0) >> QuadratureEigenstate(0, 0, 0).dual
        val1 = Coherent(0, 1) >> QuadratureEigenstate(0, np.sqrt(2 * settings.HBAR), 0).dual
        assert math.allclose(val0, val1)

    def test_wires(self):
        """Test that the wires are correct."""
        state = QuadratureEigenstate(0, 0, 0)
        for w in state.wires:
            assert w.repr == ReprEnum.QUADRATURE
