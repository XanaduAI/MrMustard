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

"""Tests for BtoQ."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import BtoQ, Coherent, GaussianKet, Identity


class TestBtoQ:
    r"""
    Tests for the ``BtoQ`` class.
    """

    def test_adjoint(self):
        btoq = BtoQ((0,), 0.5)
        adjoint_btoq = btoq.adjoint

        assert adjoint_btoq.ansatz == btoq.ansatz.conj
        assert adjoint_btoq.wires == btoq.wires.adjoint
        assert adjoint_btoq.parameters.phi == btoq.parameters.phi

    @pytest.mark.parametrize("modes", [(0,), (0, 1)])
    def test_BtoQ_QtoB(self, modes):
        component = GaussianKet.random(modes=modes)
        btoq = BtoQ(modes, 0.0)
        quad_component = component >> btoq
        new_component = quad_component >> btoq.inverse()
        A0, b0, c0 = component.ansatz.triple
        Af, bf, cf = new_component.ansatz.triple
        assert math.allclose(A0, Af)
        assert math.allclose(b0, bf)
        assert math.allclose(c0, cf)

    def test_BtoQ_with_displacement(self):
        "tests the BtoQ transformation with coherent states"

        def wavefunction_coh(alpha, quad, axis_angle):
            "alpha = x+iy of coherent state, quad is quadrature variable, axis_angle of quad axis"
            A = -1 / settings.HBAR
            b = np.exp(-1j * axis_angle) * np.sqrt(2 / settings.HBAR) * alpha
            c = (
                np.exp(-0.5 * np.abs(alpha) ** 2)
                / np.power(np.pi * settings.HBAR, 0.25)
                * np.exp(-0.5 * alpha**2 * np.exp(-2j * axis_angle))
            )
            return c * np.exp(0.5 * A * quad**2 + b * quad)

        rng = settings.get_rng()
        x = rng.random()
        y = rng.random()
        axis_angle = rng.random()
        quad = rng.random()

        state = Coherent(0, x + 1j * y)
        wavefunction = (state >> BtoQ((0,), axis_angle)).ansatz

        assert np.allclose(wavefunction(quad), wavefunction_coh(x + 1j * y, quad, axis_angle))

    def test_dual(self):
        btoq = BtoQ((0,), 0.5)
        dual_btoq = btoq.dual

        assert dual_btoq.ansatz == btoq.ansatz.conj
        assert dual_btoq.wires == btoq.wires.dual
        assert dual_btoq.parameters.phi == btoq.parameters.phi

    def test_fock_array(self):
        btoq = BtoQ((0,), 0.5)
        with pytest.raises(NotImplementedError):
            btoq.fock_array()

    def test_inverse(self):
        btoq = BtoQ((0,), 0.5)
        inv_btoq = btoq.inverse()
        assert (btoq >> inv_btoq) == Identity((0,))
