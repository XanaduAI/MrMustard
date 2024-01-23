import numpy as np
import pytest

import mrmustard.math as math
from mrmustard.physics.representations import Fock
from mrmustard.physics.ansatze import ArrayAnsatz


class TestFockRepresentation:
    r"""Tests the init and other algebras related to Fock Representation."""

    def test_init_batched(self):
        r"""Tests one can initilize a Fock representation with a batched array."""
        array = math.astensor(np.random.random((1, 5, 7, 8)))
        fock = Fock(array, batch_flag=True)
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, array)

    def test_init_non_batched(self):
        r"""Tests one can initilize a Fock representation with an array non batch, and the batch is added after init."""
        array = math.astensor(np.random.random((5, 7, 8)))
        fock = Fock(array, batch_flag=False)
        assert isinstance(fock, Fock)
        assert fock.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock.array[0, :, :, :], array)

    def test_init_from_ansatz_batched(self):
        r"""Tests one can initilize a Fock representation from ArrayAnsatz with a batched array."""
        array = math.astensor(np.random.random((4, 5, 7, 8)))
        fock = Fock.from_ansatz(ArrayAnsatz(array=array), batch_flag=True)
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, array)

    def test_init_from_ansatz_non_batched(self):
        r"""Tests one can initilize a Fock representation from ArrayAnsatz with an array non batch, and the batch is added after init."""
        array = math.astensor(np.random.random((5, 7, 8)))
        fock = Fock.from_ansatz(ArrayAnsatz(array=array), batch_flag=False)
        assert isinstance(fock, Fock)
        assert fock.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock.array[0, :, :, :], array)

    def test_and(self):
        r"""Tests the outer product of two Fock reprensentations."""
        array1 = math.astensor(np.random.random((1, 5, 7, 8)))  # where 1 is the batch.
        array2 = math.astensor(np.random.random((3, 5, 7, 8)))  # where 3 is the batch.
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (3, 5, 7, 8, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgh -> bpcdefgh", array1, array2), -1),
        )

    def test_multiply_a_scaler(self):
        r"""Tests the muplication with a scaler."""
        array1 = math.astensor(np.random.random((1, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * array1)

    def test_mul(self):
        r"""Tests the muplication of two Fock representations."""
        array1 = math.astensor(np.random.random((1, 5, 7, 8)))
        array2 = math.astensor(np.random.random((5, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock1_mul_fock2 = fock1 * fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", array1, array2), -1),
        )

    def test_divide_on_a_scaler(self):
        r"""Tests the divide on a scaler."""
        array1 = math.astensor(np.random.random((1, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, array1 / 1.5)

    def test_truedive(self):
        r"""Tests the division of two Fock representations."""
        array1 = math.astensor(np.random.random((1, 5, 7, 8)))
        array2 = math.astensor(np.random.random((5, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock1_mul_fock2 = fock1 / fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", array1, 1/array2), -1),
        )

    def test_conj(self):
        r"""Tests the conjugate of a Fock reprsentation has the correct array."""
        array = math.astensor(np.random.random((1, 5, 7, 8)))
        fock = Fock(array, batch_flag=True)
        fock_conj = fock.conj()
        assert np.allclose(fock_conj.array, np.conj(array))

    def test_matmul(self):
        r"""Tests the matmul can return the correct array."""
        array1 = math.astensor(np.random.random((2, 5, 7, 8)))
        array2 = math.astensor(np.random.random((5, 6, 7, 8, 10)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock_test = fock1[2] @ fock2[2]
        assert fock_test.array.shape == (10, 5, 7, 6, 7, 10)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgeh -> bpcdfgh", array1, array2), -1),
        )

    def test_add(self):
        r"""Tests the addition function can return the correct array."""
        array1 = math.astensor(np.random.random((2, 5, 7, 8)))
        array2 = math.astensor(np.random.random((5, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], array1[0]+array2[0])
        assert np.allclose(fock1_add_fock2.array[4], array1[0]+array2[4])
        assert np.allclose(fock1_add_fock2.array[5], array1[1]+array2[0])

    def test_sub(self):
        r"""Tests the subtraction function can return the correct array."""
        array1 = math.astensor(np.random.random((2, 5, 7, 8)))
        array2 = math.astensor(np.random.random((5, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = Fock(array2, batch_flag=True)
        fock1_sub_fock2 = fock1 - fock2
        assert fock1_sub_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_sub_fock2.array[0], array1[0]-array2[0])
        assert np.allclose(fock1_sub_fock2.array[4], array1[0]-array2[4])
        assert np.allclose(fock1_sub_fock2.array[9], array1[1]-array2[4])

    def test_trace(self):
        r"""Tests the traceo of given indices."""
        array1 = math.astensor(np.random.random((2, 5, 5, 8, 7, 6, 8, 7, 9)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = fock1.trace(idxs1=[2, 4], idxs2=[1, 7])
        assert fock2.array.shape == (2, 8, 6, 8, 9)
        assert np.allclose(fock2.array, np.einsum("bccefghfj -> beghj", array1))

    def test_reorder(self):
        r"""Tests the reorder of the array."""
        array1 = math.astensor(np.random.random((2, 5, 7, 8)))
        fock1 = Fock(array1, batch_flag=True)
        fock2 = fock1.reorder(order=(0, 3, 2, 1))
        assert fock2.array.shape == (2, 8, 7, 5)
