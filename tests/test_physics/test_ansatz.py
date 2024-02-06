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

# pylint: disable = missing-function-docstring

import numpy as np
from hypothesis import given
import pytest

from mrmustard import math
from mrmustard.physics.ansatze import PolyExpAnsatz, ArrayAnsatz
from tests.random import Abc_triple, complex_number


class TestPolyExpAnsatz:
    r"""
    Tests the ``PolyExpAnsatz`` class.
    """

    @given(Abc=Abc_triple())
    def test_init(self, Abc):
        A, b, c = Abc
        ansatz = PolyExpAnsatz(A, b, c)
        assert np.allclose(ansatz.mat[0], A)
        assert np.allclose(ansatz.vec[0], b)
        assert np.allclose(ansatz.array[0], c)

    @given(Abc1=Abc_triple(5), Abc2=Abc_triple(5))
    def test_add(self, Abc1, Abc2):
        A1, b1, c1 = Abc1
        A2, b2, c2 = Abc2

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz3 = ansatz + ansatz2

        assert np.allclose(ansatz3.mat[0], A1)
        assert np.allclose(ansatz3.vec[0], b1)
        assert np.allclose(ansatz3.array[0], c1)
        assert np.allclose(ansatz3.mat[1], A2)
        assert np.allclose(ansatz3.vec[1], b2)
        assert np.allclose(ansatz3.array[1], c2)

    @given(Abc1=Abc_triple(4), Abc2=Abc_triple(4))
    def test_mul(self, Abc1, Abc2):
        A1, b1, c1 = Abc1
        A2, b2, c2 = Abc2

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz3 = ansatz * ansatz2

        assert np.allclose(ansatz3.mat[0], A1 + A2)
        assert np.allclose(ansatz3.vec[0], b1 + b2)
        assert np.allclose(ansatz3.array[0], c1 * c2)

    @given(Abc=Abc_triple(), d=complex_number)
    def test_mul_scalar(self, Abc, d):
        A, b, c = Abc

        ansatz = PolyExpAnsatz(A, b, c)
        ansatz2 = ansatz * d

        assert np.allclose(ansatz2.mat[0], A)
        assert np.allclose(ansatz2.vec[0], b)
        assert np.allclose(ansatz2.array[0], d * c)

    @given(Abc=Abc_triple())
    def test_call(self, Abc):
        A, b, c = Abc
        ansatz = PolyExpAnsatz(A, b, c)

        assert np.allclose(ansatz(z=math.zeros_like(b)), c)

    @given(Abc1=Abc_triple(6), Abc2=Abc_triple(6))
    def test_and(self, Abc1, Abc2):
        A1, b1, c1 = Abc1
        A2, b2, c2 = Abc2

        ansatz = PolyExpAnsatz(A1, b1, c1)
        ansatz2 = PolyExpAnsatz(A2, b2, c2)
        ansatz3 = ansatz & ansatz2

        assert np.allclose(ansatz3.mat[0], math.block_diag(A1, A2))
        assert np.allclose(ansatz3.vec[0], math.concat([b1, b2], -1))
        assert np.allclose(ansatz3.array[0], c1 * c2)

    @given(Abc=Abc_triple())
    def test_eq(self, Abc):
        A, b, c = Abc

        ansatz = PolyExpAnsatz(A, b, c)
        ansatz2 = PolyExpAnsatz(2 * A, 2 * b, 2 * c)

        assert ansatz == ansatz
        assert ansatz2 == ansatz2
        assert ansatz != ansatz2
        assert ansatz2 != ansatz

    @given(Abc=Abc_triple())
    def test_simplify(self, Abc):
        """Test that we can simplify a PolyExpAnsatz object"""
        A, b, c = Abc

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

    @given(Abc=Abc_triple())
    def test_simplify_v2(self, Abc):
        A, b, c = Abc

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
            A=[np.array([[0]]), np.array([[1]])], b=[np.array([1]), np.array([0])], c=[1, 2]
        )
        ansatz._order_batch()

        assert np.allclose(ansatz.A[0], np.array([[1]]))
        assert np.allclose(ansatz.b[0], np.array([0]))
        assert ansatz.c[0] == 2
        assert np.allclose(ansatz.A[1], np.array([[0]]))
        assert np.allclose(ansatz.b[1], np.array([1]))
        assert ansatz.c[1] == 1


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
            np.array([[[[0, 0], [0, 0]], [[0, 1], [2, 3]]], [[[0, 2], [4, 6]], [[0, 3], [6, 9]]]]),
        )
        assert np.allclose(
            aa1_and_aa2.array[1],
            np.array(
                [[[[0, 0], [0, 0]], [[4, 5], [6, 7]]], [[[8, 10], [12, 14]], [[12, 15], [18, 21]]]]
            ),
        )
        assert np.allclose(
            aa1_and_aa2.array[2],
            np.array(
                [[[[0, 4], [8, 12]], [[0, 5], [10, 15]]], [[[0, 6], [12, 18]], [[0, 7], [14, 21]]]]
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
