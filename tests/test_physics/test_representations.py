# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains tests for ``Representation`` objects."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.physics.representations import Bargmann, Fock
from ..random import Abc_triple

# pylint: disable = missing-function-docstring


class TestBargmannRepresentation:
    r"""
    Tests the Bargmann Representation.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_non_batched(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple)

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_from_ansatz(self, triple):
        bargmann1 = Bargmann(*triple)
        bargmann2 = Bargmann.from_ansatz(bargmann1.ansatz)

        assert bargmann1 == bargmann2

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple).conj()

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = Bargmann(*triple1) & Bargmann(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mul(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_mul = bargmann1 * bargmann2

        assert np.allclose(bargmann_mul.A, bargmann1.A + bargmann2.A)
        assert np.allclose(bargmann_mul.b, bargmann1.b + bargmann2.b)
        assert np.allclose(bargmann_mul.c, bargmann1.c * bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_div(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_div = bargmann1 / bargmann2

        assert np.allclose(bargmann_div.A, bargmann1.A - bargmann2.A)
        assert np.allclose(bargmann_div.b, bargmann1.b - bargmann2.b)
        assert np.allclose(bargmann_div.c, bargmann1.c / bargmann2.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_add(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 + bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0))

    # def test_trace(self):
    #     bargmann = Bargmann(*Abc_triple(4)).trace([0], [2])
    #     assert np.allclose(bargmann.A.shape, (1, 2, 2))
    #     assert np.allclose(bargmann.b.shape, (1, 2))
    #     assert np.allclose(bargmann.c.shape, (1,))

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = Bargmann(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_call(self, triple):
        bargmann = Bargmann(*triple)

        assert bargmann(0.1 + 0.2j) == bargmann.ansatz(0.1 + 0.2j)


class TestFockRepresentation:
    r"""Tests the Fock Representation."""

    array578 = np.random.random((5, 7, 8))
    array1578 = np.random.random((1, 5, 7, 8))
    array2578 = np.random.random((2, 5, 7, 8))
    array5578 = np.random.random((5, 5, 7, 8))

    def test_init_batched(self):
        fock = Fock(self.array1578, batched=True)
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, self.array1578)

    def test_init_non_batched(self):
        fock = Fock(self.array578, batched=False)
        assert isinstance(fock, Fock)
        assert fock.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock.array[0, :, :, :], self.array578)

    def test_init_from_ansatz(self):
        fock1 = Fock(self.array5578)
        fock2 = Fock.from_ansatz(fock1.ansatz)
        assert fock1 == fock2

    def test_and(self):
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 5, 7, 8, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgh -> bpcdefgh", self.array1578, self.array5578), -1),
        )

    def test_multiply_a_scalar(self):
        fock1 = Fock(self.array1578, batched=True)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_mul(self):
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 * fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, self.array5578), -1),
        )

    def test_divide_on_a_scalar(self):
        fock1 = Fock(self.array1578, batched=True)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_truediv(self):
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 / fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, 1 / self.array5578), -1),
        )

    def test_conj(self):
        fock = Fock(self.array1578, batched=True)
        fock_conj = fock.conj()
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_matmul(self):
        array2 = math.astensor(np.random.random((5, 6, 7, 8, 10)))
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(array2, batched=True)
        fock_test = fock1[2] @ fock2[2]
        assert fock_test.array.shape == (10, 5, 7, 6, 7, 10)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgeh -> bpcdfgh", self.array2578, array2), -1),
        )

    def test_add(self):
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] + self.array5578[0])
        assert np.allclose(fock1_add_fock2.array[4], self.array2578[0] + self.array5578[4])
        assert np.allclose(fock1_add_fock2.array[5], self.array2578[1] + self.array5578[0])

    def test_sub(self):
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_sub_fock2 = fock1 - fock2
        assert fock1_sub_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_sub_fock2.array[0], self.array2578[0] - self.array5578[0])
        assert np.allclose(fock1_sub_fock2.array[4], self.array2578[0] - self.array5578[4])
        assert np.allclose(fock1_sub_fock2.array[9], self.array2578[1] - self.array5578[4])

    def test_trace(self):
        array1 = math.astensor(np.random.random((2, 5, 5, 1, 7, 4, 1, 7, 3)))
        fock1 = Fock(array1, batched=True)
        fock2 = fock1.trace(idxs1=[0, 3], idxs2=[1, 6])
        assert fock2.array.shape == (2, 1, 4, 1, 3)
        assert np.allclose(fock2.array, np.einsum("bccefghfj -> beghj", array1))

    def test_reorder(self):
        array1 = math.astensor(np.arange(8).reshape((1, 2, 2, 2)))
        fock1 = Fock(array1, batched=True)
        fock2 = fock1.reorder(order=(2, 1, 0))
        assert np.allclose(fock2.array, np.array([[[[0, 4], [2, 6]], [[1, 5], [3, 7]]]]))
        assert np.allclose(fock2.array, np.arange(8).reshape((1, 2, 2, 2), order="F"))
