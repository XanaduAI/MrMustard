import numpy as np
import pytest

import mrmustard.math as math
from mrmustard.physics.representations import Fock
from mrmustard.physics.ansatze import ArrayAnsatz


class TestFockRepresentation:
    r"""Tests the init and other algebras related to Fock Representation."""

    array578 = np.random.random((5, 7, 8))
    array1578 = np.random.random((1, 5, 7, 8))
    array2578 = np.random.random((2, 5, 7, 8))
    array5578 = np.random.random((5, 5, 7, 8))

    def test_init_batched(self):
        r"""Tests one can initilize a Fock representation with a batched array."""
        fock = Fock(self.array1578, batched=True)
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, self.array1578)

    def test_init_non_batched(self):
        r"""Tests one can initilize a Fock representation with a non-batched array, and the batch is added after init."""
        fock = Fock(self.array578, batched=False)
        assert isinstance(fock, Fock)
        assert fock.array.shape == (1, 5, 7, 8)
        assert np.allclose(fock.array[0, :, :, :], self.array578)

    def test_init_from_ansatz_batched(self):
        r"""Tests one can initilize a Fock representation from ArrayAnsatz with a batched array."""
        fock = Fock.from_ansatz(ArrayAnsatz(array=self.array5578))
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, self.array5578)

    def test_and(self):
        r"""Tests the outer product of two Fock reprensentations."""
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (5, 5, 7, 8, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock_test.array, -1),
            math.reshape(np.einsum("bcde, pfgh -> bpcdefgh", self.array1578, self.array5578), -1),
        )

    def test_multiply_a_scalar(self):
        r"""Tests the muplication with a scalar."""
        fock1 = Fock(self.array1578, batched=True)
        fock_test = 1.3 * fock1
        assert np.allclose(fock_test.array, 1.3 * self.array1578)

    def test_mul(self):
        r"""Tests the muplication of two Fock representations."""
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 * fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, self.array5578), -1),
        )

    def test_divide_on_a_scalar(self):
        r"""Tests the divide on a scalar."""
        fock1 = Fock(self.array1578, batched=True)
        fock_test = fock1 / 1.5
        assert np.allclose(fock_test.array, self.array1578 / 1.5)

    def test_truediv(self):
        r"""Tests the division of two Fock representations."""
        fock1 = Fock(self.array1578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_mul_fock2 = fock1 / fock2
        assert fock1_mul_fock2.array.shape == (5, 5, 7, 8)
        assert np.allclose(
            math.reshape(fock1_mul_fock2.array, -1),
            math.reshape(np.einsum("bcde, pcde -> bpcde", self.array1578, 1 / self.array5578), -1),
        )

    def test_conj(self):
        r"""Tests the conjugate of a Fock reprsentation has the correct array."""
        fock = Fock(self.array1578, batched=True)
        fock_conj = fock.conj()
        assert np.allclose(fock_conj.array, np.conj(self.array1578))

    def test_matmul(self):
        r"""Tests the matmul can return the correct array."""
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
        r"""Tests the addition function can return the correct array."""
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_add_fock2 = fock1 + fock2
        assert fock1_add_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_add_fock2.array[0], self.array2578[0] + self.array5578[0])
        assert np.allclose(fock1_add_fock2.array[4], self.array2578[0] + self.array5578[4])
        assert np.allclose(fock1_add_fock2.array[5], self.array2578[1] + self.array5578[0])

    def test_sub(self):
        r"""Tests the subtraction function can return the correct array."""
        fock1 = Fock(self.array2578, batched=True)
        fock2 = Fock(self.array5578, batched=True)
        fock1_sub_fock2 = fock1 - fock2
        assert fock1_sub_fock2.array.shape == (10, 5, 7, 8)
        assert np.allclose(fock1_sub_fock2.array[0], self.array2578[0] - self.array5578[0])
        assert np.allclose(fock1_sub_fock2.array[4], self.array2578[0] - self.array5578[4])
        assert np.allclose(fock1_sub_fock2.array[9], self.array2578[1] - self.array5578[4])

    def test_trace(self):
        r"""Tests the traceo of given indices."""
        array1 = math.astensor(np.random.random((2, 5, 5, 1, 7, 4, 1, 7, 3)))
        fock1 = Fock(array1, batched=True)
        fock2 = fock1.trace(idxs1=[0, 3], idxs2=[1, 6])
        assert fock2.array.shape == (2, 1, 4, 1, 3)
        assert np.allclose(fock2.array, np.einsum("bccefghfj -> beghj", array1))

    def test_reorder(self):
        r"""Tests the reorder of the array."""
        array1 = math.astensor(np.arange(8).reshape((1, 2, 2, 2)))
        fock1 = Fock(array1, batched=True)
        fock2 = fock1.reorder(order=(2, 1, 0))
        assert np.allclose(fock2.array, np.array([[[[0, 4], [2, 6]], [[1, 5], [3, 7]]]]))
        assert np.allclose(fock2.array, np.arange(8).reshape((1, 2, 2, 2), order="F"))
