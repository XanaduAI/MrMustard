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

# pylint: disable=fixme, missing-function-docstring, protected-access, pointless-statement

import numpy as np

from mrmustard import math, settings
from mrmustard.physics.gaussian_integrals import (
    real_gaussian_integral,
    complex_gaussian_integral_1,
    join_Abc,
    join_Abc_real,
)
from mrmustard.lab_dev import Coherent, BtoQ, Identity


class TestBtoQ:
    r"""
    Tests for the ``BtoQ`` class.
    """

    def test_adjoint(self):
        btoq = BtoQ([0], 0.5)
        adjoint_btoq = btoq.adjoint

        kets = btoq.wires.ket.indices
        assert adjoint_btoq.representation == btoq.representation.reorder(kets).conj()
        assert adjoint_btoq.wires == btoq.wires.adjoint
        assert adjoint_btoq.phi == btoq.phi
        assert isinstance(adjoint_btoq, BtoQ)

    def test_dual(self):
        btoq = BtoQ([0], 0.5)
        dual_btoq = btoq.dual

        ok = dual_btoq.wires.ket.output.indices
        ik = dual_btoq.wires.ket.input.indices
        assert dual_btoq.representation == btoq.representation.reorder(ik + ok).conj()
        assert dual_btoq.wires == btoq.wires.dual
        assert dual_btoq.phi == btoq.phi
        assert isinstance(dual_btoq, BtoQ)

    def test_inverse(self):
        btoq = BtoQ([0], 0.5)
        inv_btoq = btoq.inverse()
        assert (btoq >> inv_btoq) == Identity([0])
        assert isinstance(inv_btoq, BtoQ)

    def testBtoQ_works_correctly_by_applying_it_twice_on_a_state(self):
        A0 = np.array([[0.5, 0.3], [0.3, 0.5]]) + 0.0j
        b0 = np.zeros(2, dtype=np.complex128)
        c0 = 1.0 + 0j

        modes = [0, 1]
        BtoQ_CC1 = BtoQ(modes, 0.0)
        step1A, step1b, step1c = BtoQ_CC1.bargmann_triple(batched=False)
        Ainter, binter, cinter = complex_gaussian_integral_1(
            join_Abc((A0, b0, c0), (step1A, step1b, step1c)),
            idx_z=[0, 1],
            idx_zconj=[4, 5],
            measure=-1,
        )
        QtoBMap_CC2 = BtoQ(modes, 0.0).dual
        step2A, step2b, step2c = QtoBMap_CC2.bargmann_triple(batched=False)

        new_A, new_b, new_c = join_Abc_real(
            (Ainter[0], binter[0], cinter[0]), (step2A, step2b, step2c), [0, 1], [2, 3]
        )

        Af, bf, cf = real_gaussian_integral((new_A, new_b, new_c), idx=[0, 1])

        assert math.allclose(A0, Af)
        assert math.allclose(b0, bf)
        assert math.allclose(c0, cf)

        A0 = np.array([[0.4895454]])
        b0 = np.zeros(1)
        c0 = 1.0 + 0j

        modes = [0]
        BtoQ_CC1 = BtoQ(modes, 0.0)
        step1A, step1b, step1c = BtoQ_CC1.bargmann_triple(batched=False)
        Ainter, binter, cinter = complex_gaussian_integral_1(
            join_Abc((A0, b0, c0), (step1A, step1b, step1c)),
            idx_z=[
                0,
            ],
            idx_zconj=[2],
            measure=-1,
        )
        QtoBMap_CC2 = BtoQ(modes, 0.0).dual
        step2A, step2b, step2c = QtoBMap_CC2.bargmann_triple(batched=False)

        new_A, new_b, new_c = join_Abc_real(
            (Ainter[0], binter[0], cinter[0]), (step2A, step2b, step2c), [0], [1]
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

        x = np.random.random()
        y = np.random.random()
        axis_angle = np.random.random()
        quad = np.random.random()

        state = Coherent([0], x, y)
        wavefunction = (state >> BtoQ([0], axis_angle)).representation.ansatz

        assert np.allclose(wavefunction(quad), wavefunction_coh(x + 1j * y, quad, axis_angle))
