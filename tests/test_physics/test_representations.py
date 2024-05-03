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

from mrmustard import math, settings
from mrmustard.physics.converters import to_fock
from mrmustard.physics.triples import displacement_gate_Abc, attenuator_Abc
from mrmustard.physics.gaussian_integrals import (
    contract_two_Abc,
    complex_gaussian_integral,
)
from mrmustard.physics.representations import Bargmann, Fock
from ..random import Abc_triple

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF

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

        assert np.allclose(
            bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0)
        )
        assert np.allclose(
            bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0)
        )
        assert np.allclose(
            bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0)
        )

    def test_add_error(self):
        bargmann = Bargmann(*Abc_triple(3))
        fock = Fock(np.random.random((1, 4, 4, 4)), batched=True)

        with pytest.raises(ValueError):
            bargmann + fock

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(
            bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0)
        )
        assert np.allclose(
            bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0)
        )
        assert np.allclose(
            bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0)
        )

    def test_trace(self):
        triple = Abc_triple(4)
        bargmann = Bargmann(*triple).trace([0], [2])
        A, b, c = complex_gaussian_integral(triple, [0], [2])

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = Bargmann(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_call(self, triple):
        bargmann = Bargmann(*triple)

        assert bargmann(0.1 + 0.2j) == bargmann.ansatz(0.1 + 0.2j)

    def test_matmul_barg_barg(self):
        triple1 = Abc_triple(3)
        triple2 = Abc_triple(3)

        res1 = Bargmann(*triple1) @ Bargmann(*triple2)
        exp1 = contract_two_Abc(triple1, triple2, [], [])
        assert np.allclose(res1.A, exp1[0])
        assert np.allclose(res1.b, exp1[1])
        assert np.allclose(res1.c, exp1[2])

    @pytest.mark.parametrize("n1", [1, 2])
    @pytest.mark.parametrize("n2", [1, 2])
    def test_matmul_barg_fock(self, n1, n2):
        settings.AUTOCUTOFF_MAX_CUTOFF = 3

        d01_barg = Bargmann(*displacement_gate_Abc([0.1, 0.2]))
        a0_barg = Bargmann(*attenuator_Abc(0.7))

        d01_barg = d01_barg if n1 == 1 else d01_barg + d01_barg
        a0_barg = a0_barg if n2 == 1 else a0_barg + a0_barg

        a0_fock = to_fock(a0_barg, shape=(2, 3, 4, 5))

        res = (d01_barg[0] @ a0_fock[0]).array
        assert res.shape == (n1 * n2, 3, 3, 3, 3, 4, 5)

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0


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

    def test_sum_batch(self):
        fock = Fock(self.array2578, batched=True)
        fock_collapsed = fock.sum_batch()[0]
        assert fock_collapsed.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock_collapsed.array, np.sum(self.array2578, axis=0))

    def test_and(self):
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 5, 7, 8, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(
                np.einsum("bcde, pfgh -> bpcdefgh", self.array1578, self.array5578), -1
            ),
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
            math.reshape(
                np.einsum("bcde, pcde -> bpcde", self.array1578, self.array5578), -1
            ),
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
            math.reshape(
                np.einsum("bcde, pcde -> bpcde", self.array1578, 1 / self.array5578), -1
            ),
        )

    def test_conj(self):
        fock = Fock(self.array1578, batched=True)
        fock_conj = fock.conj()
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_matmul_fock_fock(self):
        array2 = math.astensor(np.random.random((5, 6, 7, 8, 10)))
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(array2, batched=True)
        fock_test = fock1[2] @ fock2[2]
        assert fock_test.array.shape == (10, 5, 7, 6, 7, 10)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(
                np.einsum("bcde, pfgeh -> bpcdfgh", self.array2578, array2), -1
            ),
        )

    @pytest.mark.parametrize("n1", [1, 2])
    @pytest.mark.parametrize("n2", [1, 2])
    def test_matmul_fock_barg(self, n1, n2):
        settings.AUTOCUTOFF_MAX_CUTOFF = 3

        d01_barg = Bargmann(*displacement_gate_Abc([0.1, 0.2]))
        a0_barg = Bargmann(*attenuator_Abc(0.7))

        d01_barg = d01_barg if n1 == 1 else d01_barg + d01_barg
        a0_barg = a0_barg if n2 == 1 else a0_barg + a0_barg

        d01_fock = to_fock(d01_barg, shape=(2, 3, 4, 5))
        a0_fock = to_fock(a0_barg, shape=(2, 3, 4, 5))

        assert (d01_fock[0] @ a0_barg[0]).array.shape == (n1 * n2, 3, 4, 5, 3, 3, 3)
        assert (d01_fock[0] @ a0_fock[0]).array.shape == (n1 * n2, 3, 4, 5, 3, 4, 5)

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0

    def test_add(self):
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(
            fock1_add_fock2.array[0], self.array2578[0] + self.array5578[0]
        )
        assert np.allclose(
            fock1_add_fock2.array[4], self.array2578[0] + self.array5578[4]
        )
        assert np.allclose(
            fock1_add_fock2.array[5], self.array2578[1] + self.array5578[0]
        )

    def test_sub(self):
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_sub_fock2 = fock1 - fock2
        assert fock1_sub_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(
            fock1_sub_fock2.array[0], self.array2578[0] - self.array5578[0]
        )
        assert np.allclose(
            fock1_sub_fock2.array[4], self.array2578[0] - self.array5578[4]
        )
        assert np.allclose(
            fock1_sub_fock2.array[9], self.array2578[1] - self.array5578[4]
        )

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
        assert np.allclose(
            fock2.array, np.array([[[[0, 4], [2, 6]], [[1, 5], [3, 7]]]])
        )
        assert np.allclose(fock2.array, np.arange(8).reshape((1, 2, 2, 2), order="F"))

    @pytest.mark.parametrize("batched", [True, False])
    def test_reduce(self, batched):
        shape = (1, 3, 3, 3) if batched else (3, 3, 3)
        array1 = math.astensor(np.arange(27).reshape(shape))
        fock1 = Fock(array1, batched=batched)

        fock2 = fock1.reduce(3)
        assert fock1 == fock2

        fock3 = fock1.reduce(2)
        array3 = math.astensor([[[0, 1], [3, 4]], [[9, 10], [12, 13]]])
        assert fock3 == Fock(array3)

        fock4 = fock1.reduce((2, 1, 3, 1))
        array4 = math.astensor([[[0], [3], [6]]])
        assert fock4 == Fock(array4)

    def test_reduce_error(self):
        array1 = math.astensor(np.arange(27).reshape((3, 3, 3)))
        fock1 = Fock(array1)

        with pytest.raises(ValueError, match="Expected ``shape``"):
            fock1.reduce((1, 2))

        with pytest.raises(ValueError, match="Expected ``shape``"):
            fock1.reduce((1, 2, 3, 4, 5))
