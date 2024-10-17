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

"""This module contains tests for ``Ansatz`` objects."""

# pylint: disable = missing-function-docstring, pointless-statement, comparison-with-itself

import numpy as np
import pytest

from mrmustard import math
from mrmustard.physics.ansatze import (
    PolyExpAnsatz,
    ArrayAnsatz,
    bargmann_Abc_to_phasespace_cov_means,
)
from mrmustard.lab_dev.states.dm import DM
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.lab_dev.circuit_components_utils import BtoPS
from ..random import Abc_triple


class TestArrayAnsatz:
    r"""Tests all algebra related to ArrayAnsatz."""

    def test_init_(self):
        array = np.random.random((2, 4, 5))
        aa = ArrayAnsatz(array=array)
        assert isinstance(aa, ArrayAnsatz)
        assert np.allclose(aa.array, array)

    def test_neg(self):
        array = np.random.random((2, 4, 5))
        aa = ArrayAnsatz(array=array)
        minusaa = -aa
        assert isinstance(minusaa, ArrayAnsatz)
        assert np.allclose(minusaa.array, -array)

    def test_equal(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array)
        assert aa1 == aa2

    def test_add(self):
        array = np.arange(8).reshape(2, 2, 2)
        array2 = np.arange(8).reshape(2, 2, 2)
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)
        aa1_add_aa2 = aa1 + aa2

        assert isinstance(aa1_add_aa2, ArrayAnsatz)
        assert aa1_add_aa2.array.shape == (4, 2, 2)
        assert np.allclose(aa1_add_aa2.array[0], np.array([[0, 2], [4, 6]]))
        assert np.allclose(aa1_add_aa2.array[1], np.array([[4, 6], [8, 10]]))
        assert np.allclose(aa1_add_aa2.array[2], np.array([[4, 6], [8, 10]]))
        assert np.allclose(aa1_add_aa2.array[3], np.array([[8, 10], [12, 14]]))

        array3 = np.arange(2).reshape(2, 1, 1)
        aa3 = ArrayAnsatz(array=array3)
        aa2_add_aa3 = aa2 + aa3

        assert isinstance(aa2_add_aa3, ArrayAnsatz)
        assert np.allclose(aa2_add_aa3.array[0], np.array([[0, 1], [2, 3]]))
        assert np.allclose(aa2_add_aa3.array[1], np.array([[1, 1], [2, 3]]))
        assert np.allclose(aa2_add_aa3.array[2], np.array([[4, 5], [6, 7]]))
        assert np.allclose(aa2_add_aa3.array[3], np.array([[5, 5], [6, 7]]))

        array4 = np.array([1]).reshape(1, 1, 1)
        aa4 = ArrayAnsatz(array=array4)
        aa2_add_aa4 = aa2 + aa4

        assert isinstance(aa2_add_aa4, ArrayAnsatz)
        assert np.allclose(aa2_add_aa4.array[0], np.array([[1, 1], [2, 3]]))
        assert np.allclose(aa2_add_aa4.array[1], np.array([[5, 5], [6, 7]]))

    def test_and(self):
        array = np.arange(8).reshape(2, 2, 2)
        array2 = np.arange(8).reshape(2, 2, 2)
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)
        aa1_and_aa2 = aa1 & aa2
        assert isinstance(aa1_and_aa2, ArrayAnsatz)
        assert aa1_and_aa2.array.shape == (4, 2, 2, 2, 2)
        assert np.allclose(
            aa1_and_aa2.array[0],
            np.array(
                [
                    [[[0, 0], [0, 0]], [[0, 1], [2, 3]]],
                    [[[0, 2], [4, 6]], [[0, 3], [6, 9]]],
                ]
            ),
        )
        assert np.allclose(
            aa1_and_aa2.array[1],
            np.array(
                [
                    [[[0, 0], [0, 0]], [[4, 5], [6, 7]]],
                    [[[8, 10], [12, 14]], [[12, 15], [18, 21]]],
                ]
            ),
        )
        assert np.allclose(
            aa1_and_aa2.array[2],
            np.array(
                [
                    [[[0, 4], [8, 12]], [[0, 5], [10, 15]]],
                    [[[0, 6], [12, 18]], [[0, 7], [14, 21]]],
                ]
            ),
        )
        assert np.allclose(
            aa1_and_aa2.array[3],
            np.array(
                [
                    [[[16, 20], [24, 28]], [[20, 25], [30, 35]]],
                    [[[24, 30], [36, 42]], [[28, 35], [42, 49]]],
                ]
            ),
        )

    def test_mul_a_scalar(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 * 8
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array * 8)

    def test_mul(self):
        array = np.arange(8).reshape(2, 2, 2)
        array2 = np.arange(8).reshape(2, 2, 2)
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)
        aa1_mul_aa2 = aa1 * aa2
        assert isinstance(aa1_mul_aa2, ArrayAnsatz)
        assert aa1_mul_aa2.array.shape == (4, 2, 2)
        assert np.allclose(aa1_mul_aa2.array[0], np.array([[0, 1], [4, 9]]))
        assert np.allclose(aa1_mul_aa2.array[1], np.array([[0, 5], [12, 21]]))
        assert np.allclose(aa1_mul_aa2.array[2], np.array([[0, 5], [12, 21]]))
        assert np.allclose(aa1_mul_aa2.array[3], np.array([[16, 25], [36, 49]]))

        array3 = np.arange(2).reshape(2, 1, 1)
        aa3 = ArrayAnsatz(array=array3)
        aa2_mul_aa3 = aa2 * aa3
        assert isinstance(aa2_mul_aa3, ArrayAnsatz)
        assert aa2_mul_aa3.array.shape == (4, 2, 2)
        assert np.allclose(aa2_mul_aa3.array[0], np.array([[0, 0], [0, 0]]))
        assert np.allclose(aa2_mul_aa3.array[1], np.array([[0, 0], [0, 0]]))
        assert np.allclose(aa2_mul_aa3.array[2], np.array([[0, 0], [0, 0]]))
        assert np.allclose(aa2_mul_aa3.array[3], np.array([[4, 0], [0, 0]]))

        array4 = np.array([1]).reshape(1, 1, 1)
        aa4 = ArrayAnsatz(array=array4)
        aa2_mul_aa4 = aa2 * aa4

        assert isinstance(aa2_mul_aa4, ArrayAnsatz)
        assert np.allclose(aa2_mul_aa4.array[0], np.array([[0, 0], [0, 0]]))
        assert np.allclose(aa2_mul_aa4.array[1], np.array([[4, 0], [0, 0]]))

    def test_truediv_a_scalar(self):
        array = np.random.random((2, 4, 5))
        aa1 = ArrayAnsatz(array=array)
        aa1_scalar = aa1 / 6
        assert isinstance(aa1_scalar, ArrayAnsatz)
        assert np.allclose(aa1_scalar.array, array / 6)

    def test_div(self):
        array = np.arange(9)[1:].reshape(2, 2, 2)
        array2 = np.arange(9)[1:].reshape(2, 2, 2)
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)
        aa1_div_aa2 = aa1 / aa2
        assert isinstance(aa1_div_aa2, ArrayAnsatz)
        assert aa1_div_aa2.array.shape == (4, 2, 2)
        assert np.allclose(aa1_div_aa2.array[0], np.array([[1.0, 1.0], [1.0, 1.0]]))
        assert np.allclose(aa1_div_aa2.array[1], np.array([[0.2, 0.33333], [0.42857143, 0.5]]))
        assert np.allclose(aa1_div_aa2.array[2], np.array([[5.0, 3.0], [2.33333333, 2.0]]))
        assert np.allclose(aa1_div_aa2.array[3], np.array([[1.0, 1.0], [1.0, 1.0]]))

        array3 = np.arange(3)[1:].reshape(2, 1, 1)
        aa3 = ArrayAnsatz(array=array3)
        aa3_div_aa2 = aa3 / aa2
        assert isinstance(aa3_div_aa2, ArrayAnsatz)
        assert aa3_div_aa2.array.shape == (4, 2, 2)
        assert np.allclose(aa3_div_aa2.array[0], np.array([[1, 0], [0, 0]]))
        assert np.allclose(aa3_div_aa2.array[1], np.array([[0.2, 0], [0, 0]]))
        assert np.allclose(aa3_div_aa2.array[2], np.array([[2, 0], [0, 0]]))
        assert np.allclose(aa3_div_aa2.array[3], np.array([[0.4, 0], [0, 0]]))

        array4 = np.array([2]).reshape(1, 1, 1)
        aa4 = ArrayAnsatz(array=array4)
        aa4_div_aa2 = aa4 / aa2

        assert isinstance(aa4_div_aa2, ArrayAnsatz)
        assert np.allclose(aa4_div_aa2.array[0], np.array([[2, 0], [0, 0]]))
        assert np.allclose(aa4_div_aa2.array[1], np.array([[0.4, 0], [0, 0]]))

    def test_algebra_with_different_shape_of_array_raise_errors(self):
        array = np.random.random((2, 4, 5))
        array2 = np.random.random((3, 4, 8, 9))
        aa1 = ArrayAnsatz(array=array)
        aa2 = ArrayAnsatz(array=array2)

        with pytest.raises(Exception):
            aa1 + aa2

        with pytest.raises(Exception):
            aa1 - aa2

        with pytest.raises(Exception):
            aa1 * aa2

        with pytest.raises(Exception):
            aa1 / aa2

        with pytest.raises(Exception):
            aa1 == aa2

    def test_bargmann_Abc_to_phasespace_cov_means(self):
        # The init state cov and means comes from the random state 'state = Gaussian(1) >> Dgate([0.2], [0.3])'
        state_cov = np.array([[0.32210229, -0.99732956], [-0.99732956, 6.1926484]])
        state_means = np.array([0.2, 0.3])
        state = DM.from_bargmann([0], wigner_to_bargmann_rho(state_cov, state_means))
        state_after = state >> BtoPS(modes=[0], s=0)  # pylint: disable=protected-access
        A1, b1, c1 = state_after.bargmann_triple()
        (
            new_state_cov,
            new_state_means,
            new_state_coeff,
        ) = bargmann_Abc_to_phasespace_cov_means(A1, b1, c1)
        assert np.allclose(state_cov, new_state_cov)
        assert np.allclose(state_means, new_state_means)
        assert np.allclose(1.0, new_state_coeff)

        state_cov = np.array(
            [
                [1.00918303, -0.33243548, 0.15202393, -0.07540124],
                [-0.33243548, 1.2203162, -0.03961978, 0.30853472],
                [0.15202393, -0.03961978, 1.11158673, 0.28786279],
                [-0.07540124, 0.30853472, 0.28786279, 0.97833402],
            ]
        )
        state_means = np.array([0.4, 0.6, 0.0, 0.0])
        A, b, c = wigner_to_bargmann_rho(state_cov, state_means)
        state = DM.from_bargmann(modes=[0, 1], triple=(A, b, c))

        state_after = state >> BtoPS(modes=[0, 1], s=0)  # pylint: disable=protected-access
        A1, b1, c1 = state_after.bargmann_triple()
        (
            new_state_cov1,
            new_state_means1,
            new_state_coeff1,
        ) = bargmann_Abc_to_phasespace_cov_means(A1, b1, c1)

        A22, b22, c22 = (state >> BtoPS([0], 0) >> BtoPS([1], 0)).bargmann_triple()
        (
            new_state_cov22,
            new_state_means22,
            new_state_coeff22,
        ) = bargmann_Abc_to_phasespace_cov_means(A22, b22, c22)
        assert math.allclose(new_state_cov22, state_cov)
        assert math.allclose(new_state_cov1, state_cov)
        assert math.allclose(new_state_means1, state_means)
        assert math.allclose(new_state_means22, state_means)
        assert math.allclose(new_state_coeff1, 1.0)
        assert math.allclose(new_state_coeff22, 1.0)


class TestPolyExpAnsatz:
    r"""
    Tests the ``PolyExpAnsatz`` class.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init(self, triple):
        A, b, c = triple
        ansatz = PolyExpAnsatz(A, b, c)

        assert np.allclose(ansatz.mat[0], A)
        assert np.allclose(ansatz.vec[0], b)
        assert np.allclose(ansatz.array[0], c)

    def test_add(self):
        A1, b1, _ = Abc_triple(5)
        c1 = np.random.random(size=(1, 3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.random(size=(1, 2, 2))

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)

        ansatz3 = ansatz + ansatz2

        assert np.allclose(ansatz3.mat[0], A1)
        assert np.allclose(ansatz3.vec[0], b1)
        assert np.allclose(ansatz3.array[0], c1[0])
        assert np.allclose(ansatz3.mat[1], A2)
        assert np.allclose(ansatz3.vec[1], b2)
        assert np.allclose(ansatz3.array[1][:2, :2], c2[0])

    def test_mul(self):
        A1, b1, _ = Abc_triple(2)
        c1 = np.random.random(size=(1, 4))
        A2, b2, _ = Abc_triple(2)
        c2 = np.random.random(size=(1, 4))

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz3 = ansatz * ansatz2

        A3 = np.block(
            [
                [
                    A1[:1, :1] + A2[:1, :1],
                    A1[:1, 1:],
                    A2[:1, 1:],
                ],
                [
                    A1[1:, :1],
                    A1[1:, 1:],
                    math.zeros((1, 1), dtype=np.complex128),
                ],
                [
                    A2[1:, :1],
                    np.zeros((1, 1), dtype=np.complex128),
                    A2[1:, 1:],
                ],
            ]
        )
        b3 = np.concatenate((b1[:1] + b2[:1], b1[1:], b2[1:]))
        c3 = np.outer(c1, c2).reshape(4, 4)
        assert np.allclose(ansatz3.mat[0], A3)
        assert np.allclose(ansatz3.vec[0], b3)
        assert np.allclose(ansatz3.array[0], c3)

    def test_mul_scalar(self):
        A, b, c = Abc_triple(5)
        d = 0.1

        ansatz = PolyExpAnsatz(A, b, c)

        ansatz2 = ansatz * d

        assert np.allclose(ansatz2.mat[0], A)
        assert np.allclose(ansatz2.vec[0], b)
        assert np.allclose(ansatz2.array[0], d * c)

    def test_truediv(self):
        A1, b1, c1 = Abc_triple(5)
        A2, b2, c2 = Abc_triple(5)

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)

        ansatz3 = ansatz / ansatz2

        assert np.allclose(ansatz3.mat[0], A1 - A2)
        assert np.allclose(ansatz3.vec[0], b1 - b2)
        assert np.allclose(ansatz3.array[0], c1 / c2)

    def test_truediv_scalar(self):
        A, b, c = Abc_triple(5)
        d = 0.1

        ansatz = PolyExpAnsatz(A, b, c)

        ansatz2 = ansatz / d

        assert np.allclose(ansatz2.mat[0], A)
        assert np.allclose(ansatz2.vec[0], b)
        assert np.allclose(ansatz2.array[0], c / d)

    def test_call(self):
        A, b, c = Abc_triple(5)
        ansatz = PolyExpAnsatz(A, b, c)

        assert np.allclose(ansatz(z=math.zeros_like(b)), c)

        A, b, _ = Abc_triple(4)
        c = np.random.random(size=(1, 3, 3, 3))
        ansatz = PolyExpAnsatz(A, b, c)
        z = np.random.uniform(-10, 10, size=(7, 2))
        with pytest.raises(
            Exception, match="The sum of the dimension of the argument and polynomial"
        ):
            ansatz(z)

        A = np.array([[0, 1], [1, 0]])
        b = np.zeros(2)
        c = c = np.zeros(10, dtype=complex).reshape(1, -1)
        c[0, -1] = 1
        obj1 = PolyExpAnsatz(A, b, c)

        nine_factorial = np.prod(np.arange(1, 9))
        assert np.allclose(obj1(np.array([[0.1]])), 0.1**9 / np.sqrt(nine_factorial))

    def test_and(self):
        A1, b1, _ = Abc_triple(6)
        c1 = np.random.random(size=(1, 4, 4))
        A2, b2, _ = Abc_triple(6)
        c2 = np.random.random(size=(1, 4, 4))

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)

        ansatz3 = ansatz & ansatz2

        A3 = np.block(
            [
                [
                    A1[:4, :4],
                    np.zeros((4, 4), dtype=complex),
                    A1[:4, 4:],
                    np.zeros((4, 2), dtype=complex),
                ],
                [
                    np.zeros((4, 4), dtype=complex),
                    A2[:4:, :4],
                    np.zeros((4, 2), dtype=complex),
                    A2[:4, 4:],
                ],
                [
                    A1[4:, :4],
                    np.zeros((2, 4), dtype=complex),
                    A1[4:, 4:],
                    np.zeros((2, 2), dtype=complex),
                ],
                [
                    np.zeros((2, 4), dtype=complex),
                    A2[4:, :4],
                    np.zeros((2, 2), dtype=complex),
                    A2[4:, 4:],
                ],
            ]
        )
        b3 = np.concatenate((b1[:4], b2[:4], b1[4:], b2[4:]))
        c3 = np.outer(c1, c2).reshape(4, 4, 4, 4)
        assert np.allclose(ansatz3.mat[0], A3)
        assert np.allclose(ansatz3.vec[0], b3)
        assert np.allclose(ansatz3.array[0], c3)

    def test_eq(self):
        A, b, c = Abc_triple(5)

        ansatz = PolyExpAnsatz(A, b, c)
        ansatz2 = PolyExpAnsatz(2 * A, 2 * b, 2 * c)

        assert ansatz == ansatz
        assert ansatz2 == ansatz2
        assert ansatz != ansatz2
        assert ansatz2 != ansatz

    def test_simplify(self):
        A, b, c = Abc_triple(5)

        ansatz = PolyExpAnsatz(A, b, c)

        ansatz = ansatz + ansatz

        assert np.allclose(ansatz.A[0], ansatz.A[1])
        assert np.allclose(ansatz.A[0], A)
        assert np.allclose(ansatz.b[0], ansatz.b[1])
        assert np.allclose(ansatz.b[0], b)

        ansatz.simplify()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert ansatz.c == 2 * c

    def test_simplify_v2(self):
        A, b, c = Abc_triple(5)

        ansatz = PolyExpAnsatz(A, b, c)

        ansatz = ansatz + ansatz

        assert np.allclose(ansatz.A[0], ansatz.A[1])
        assert np.allclose(ansatz.A[0], A)
        assert np.allclose(ansatz.b[0], ansatz.b[1])
        assert np.allclose(ansatz.b[0], b)

        ansatz.simplify_v2()
        assert len(ansatz.A) == 1
        assert len(ansatz.b) == 1
        assert np.allclose(ansatz.c, 2 * c)

    def test_order_batch(self):
        ansatz = PolyExpAnsatz(
            A=[np.array([[0]]), np.array([[1]])],
            b=[np.array([1]), np.array([0])],
            c=[1, 2],
        )
        ansatz._order_batch()  # pylint: disable=protected-access

        assert np.allclose(ansatz.A[0], np.array([[1]]))
        assert np.allclose(ansatz.b[0], np.array([0]))
        assert ansatz.c[0] == 2
        assert np.allclose(ansatz.A[1], np.array([[0]]))
        assert np.allclose(ansatz.b[1], np.array([1]))
        assert ansatz.c[1] == 1

    def test_polynomial_shape(self):
        A, b, _ = Abc_triple(4)
        c = np.array([[1, 2, 3]])
        ansatz = PolyExpAnsatz(A, b, c)

        poly_dim, poly_shape = ansatz.polynomial_shape
        assert np.allclose(poly_dim, 1)
        assert np.allclose(poly_shape, (3,))

        A1, b1, _ = Abc_triple(4)
        c1 = np.array([[1, 2, 3]])
        ansatz1 = PolyExpAnsatz(A1, b1, c1)

        A2, b2, _ = Abc_triple(4)
        c2 = np.array([[1, 2, 3]])
        ansatz2 = PolyExpAnsatz(A2, b2, c2)

        ansatz3 = ansatz1 * ansatz2

        poly_dim, poly_shape = ansatz3.polynomial_shape
        assert np.allclose(poly_dim, 2)
        assert np.allclose(poly_shape, (3, 3))

    def test_decompose_ansatz(self):
        A, b, _ = Abc_triple(4)
        c = np.random.uniform(-10, 10, size=(1, 3, 3, 3))
        ansatz = PolyExpAnsatz(A, b, c)

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(1, 1))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (1, 2, 2))

    def test_decompose_ansatz_batch(self):
        """
        In this test the batch dimension of both ``z`` and ``Abc`` is tested.
        """
        A1, b1, _ = Abc_triple(4)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3))
        A2, b2, _ = Abc_triple(4)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3))
        ansatz = PolyExpAnsatz([A1, A2], [b1, b2], [c1, c2])

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(3, 1))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 2, 2))
        assert np.allclose(decomp_ansatz.b.shape, (2, 2))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9))

        A1, b1, _ = Abc_triple(5)
        c1 = np.random.uniform(-10, 10, size=(3, 3, 3))
        A2, b2, _ = Abc_triple(5)
        c2 = np.random.uniform(-10, 10, size=(3, 3, 3))
        ansatz = PolyExpAnsatz([A1, A2], [b1, b2], [c1, c2])

        decomp_ansatz = ansatz.decompose_ansatz()
        z = np.random.uniform(-10, 10, size=(3, 2))
        assert np.allclose(ansatz(z), decomp_ansatz(z))
        assert np.allclose(decomp_ansatz.A.shape, (2, 4, 4))
        assert np.allclose(decomp_ansatz.b.shape, (2, 4))
        assert np.allclose(decomp_ansatz.c.shape, (2, 9, 9))

    def test_call_none(self):
        A1, b1, _ = Abc_triple(7)
        A2, b2, _ = Abc_triple(7)
        A3, b3, _ = Abc_triple(7)

        batch = 3
        c = np.random.random(size=(batch, 5, 5, 5)) / 1000

        obj = PolyExpAnsatz([A1, A2, A3], [b1, b2, b3], c)
        z0 = np.array([[None, 2, None, 5]])
        z1 = np.array([[1, 2, 4, 5]])
        z2 = np.array([[1, 4]])
        obj_none = obj(z0)
        val1 = obj(z1)
        val2 = obj_none(z2)
        assert np.allclose(val1, val2)

        obj1 = PolyExpAnsatz(A1, b1, c[0].reshape(1, 5, 5, 5))
        z0 = np.array([[None, 2, None, 5], [None, 1, None, 4]])
        z1 = np.array([[1, 2, 4, 5], [2, 1, 4, 4]])
        z2 = np.array([[1, 4], [2, 4]])
        obj1_none = obj1(z0)
        obj1_none0 = PolyExpAnsatz(
            obj1_none.A[0], obj1_none.b[0], obj1_none.c[0].reshape(1, 5, 5, 5)
        )
        obj1_none1 = PolyExpAnsatz(
            obj1_none.A[1], obj1_none.b[1], obj1_none.c[1].reshape(1, 5, 5, 5)
        )
        val1 = obj1(z1)
        val2 = np.array(
            (obj1_none0(z2[0].reshape(1, -1)), obj1_none1(z2[1].reshape(1, -1)))
        ).reshape(-1)
        assert np.allclose(val1, val2)

    def test_add_different_poly_wires(self):
        "tests that A and b are padded correctly"
        A1 = np.random.random((1, 2, 2))
        A2 = np.random.random((1, 3, 3))
        b1 = np.random.random((1, 2))
        b2 = np.random.random((1, 3))
        c1 = np.random.random((1,))
        c2 = np.random.random((1, 11))
        ansatz1 = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz_sum = ansatz1 + ansatz2
        assert ansatz_sum.mat.shape == (2, 3, 3)
        assert ansatz_sum.vec.shape == (2, 3)
        assert ansatz_sum.array.shape == (2, 11)
        ansatz_sum = ansatz2 + ansatz1
        assert ansatz_sum.mat.shape == (2, 3, 3)
        assert ansatz_sum.vec.shape == (2, 3)
        assert ansatz_sum.array.shape == (2, 11)

    def test_inconsistent_poly_shapes(self):
        A1 = np.random.random((1, 2, 2))
        A2 = np.random.random((1, 3, 3))
        b1 = np.random.random((1, 2))
        b2 = np.random.random((1, 3))
        c1 = np.random.random((1,))
        c2 = np.random.random((1, 5, 11))
        ansatz1 = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        with pytest.raises(ValueError):
            ansatz1 + ansatz2  # pylint: disable=pointless-statement
