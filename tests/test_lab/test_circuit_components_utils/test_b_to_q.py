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
from mrmustard.lab import BtoQ, Coherent, Identity
from mrmustard.physics.gaussian_integrals import (
    complex_gaussian_integral_1,
    join_Abc,
    join_Abc_real,
    real_gaussian_integral,
)


class TestBtoQ:
    r"""
    Tests for the ``BtoQ`` class.
    """

    def test_adjoint(self):
        btoq = BtoQ((0,), 0.5)
        adjoint_btoq = btoq.adjoint

        kets = btoq.wires.ket.indices
        assert adjoint_btoq.ansatz == btoq.ansatz.reorder(kets).conj
        assert adjoint_btoq.wires == btoq.wires.adjoint
        assert adjoint_btoq.parameters.phi == btoq.parameters.phi

    def test_BtoQ_twice_on_a_state(self):
        A0 = math.astensor([[0.5, 0.3], [0.3, 0.5]]) + 0.0j
        b0 = math.zeros(2, dtype=np.complex128)
        c0 = math.astensor(1.0 + 0.0j)

        modes = (0, 1)
        BtoQ_CC1 = BtoQ(modes, 0.0)
        step1A, step1b, step1c = BtoQ_CC1.bargmann_triple()
        Ainter, binter, cinter = complex_gaussian_integral_1(
            join_Abc((A0, b0, c0), (step1A, step1b, step1c)),
            idx_z=[0, 1],
            idx_zconj=[4, 5],
            measure=-1,
        )
        QtoBMap_CC2 = BtoQ(modes, 0.0).dual
        step2A, step2b, step2c = QtoBMap_CC2.bargmann_triple()

        new_A, new_b, new_c = join_Abc_real(
            (Ainter, binter, cinter),
            (step2A, step2b, step2c),
            [0, 1],
            [2, 3],
        )

        Af, bf, cf = real_gaussian_integral((new_A, new_b, new_c), idx=[0, 1])

        assert math.allclose(A0, Af)
        assert math.allclose(b0, bf)
        assert math.allclose(c0, cf)

        A0 = math.astensor([[0.4895454]])
        b0 = math.zeros(1)
        c0 = math.astensor(1.0 + 0.0j)

        modes = (0,)
        BtoQ_CC1 = BtoQ(modes, 0.0)
        step1A, step1b, step1c = BtoQ_CC1.bargmann_triple()
        Ainter, binter, cinter = complex_gaussian_integral_1(
            join_Abc((A0, b0, c0), (step1A, step1b, step1c)),
            idx_z=[
                0,
            ],
            idx_zconj=[2],
            measure=-1,
        )
        QtoBMap_CC2 = BtoQ(modes, 0.0).dual
        step2A, step2b, step2c = QtoBMap_CC2.bargmann_triple()

        new_A, new_b, new_c = join_Abc_real(
            (Ainter, binter, cinter),
            (step2A, step2b, step2c),
            [0],
            [1],
        )

        Af, bf, cf = real_gaussian_integral((new_A, new_b, new_c), idx=[0])

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

        rng = settings.rng
        x = rng.random()
        y = rng.random()
        axis_angle = rng.random()
        quad = rng.random()

        state = Coherent(0, x, y)
        wavefunction = (state >> BtoQ((0,), axis_angle)).ansatz

        assert np.allclose(wavefunction(quad), wavefunction_coh(x + 1j * y, quad, axis_angle))

    def test_dual(self):
        btoq = BtoQ((0,), 0.5)
        dual_btoq = btoq.dual

        ok = dual_btoq.wires.ket.output.indices
        ik = dual_btoq.wires.ket.input.indices
        assert dual_btoq.ansatz == btoq.ansatz.reorder(ik + ok).conj
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
