import numpy as np
import pytest

import mrmustard.math as math
from mrmustard.physics.representations import Fock
from mrmustard.physics.ansatze import ArrayAnsatz


class TestFockRepresentation:
    r"""Tests the init and other algebras related to Fock Representation."""
    def test_init(self):
        r"""Tests one can initilize a Fock representation."""
        array = math.astensor(np.random.random((1,5,7,8)))
        fock = Fock(array)
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, array)

    def test_init_from_ansatz(self):
        r"""Tests one can initilize a Fock representation from ArrayAnsatz."""
        array = math.astensor(np.random.random((1,5,7,8)))
        fock = Fock.from_ansatz(ArrayAnsatz(array=array))
        assert isinstance(fock, Fock)
        assert np.allclose(fock.array, array)

    def test_outer_product(self):
        r"""Tests the outer product of two Fock reprensentations."""
        array1 = math.astensor(np.random.random((1,5,7,8))) # where 1 is the batch.
        array2 = math.astensor(np.random.random((3,5,7,8))) # where 3 is the batch.
        fock1 = Fock(array1)
        fock2 = Fock(array2)
        fock_test = fock1 & fock2
        assert fock_test.array.shape == (3, 5, 7, 8, 5, 7, 8)
        assert np.allclose(fock_test.array, np.einsum("bcde, bfgh -> bcdefgh"))
    
    def test_multiply_a_scaler(self):
        r"""Tests the muplication with a scaler."""
        array1 = math.astensor(np.random.random((1,5,7,8))) # where 1 is the batch.
        array2 = math.astensor(np.random.random((1,5,7,8))) # where 1 is the batch.
        fock1 = Fock(array1)
        fock2 = Fock(array2)
        fock_test = 1.3 * fock1 - fock2 * 2.1
        assert np.allclose(fock_test, 1.3*array1 - 2.1*array2)
    
    def test_divide_on_a_scaler(self):
        r"""Tests the divide on a scaler."""
        array1 = math.astensor(np.random.random((1,5,7,8))) # where 1 is the batch.
        fock1 = Fock(array1)
        fock_test = fock1/1.5
        assert np.allclose(fock_test, array1/1.5)

    def test_conj(self):
        r"""Tests the conjugate of a Fock reprsentation has the correct array."""
        array = math.astensor(np.random.random((1,5,7,8)))
        fock = Fock(array)
        fock_conj = fock.conj()
        assert np.allclose(fock_conj.arrat, np.conj(array))

    def test_matmul(self):
        r"""Tests the matmul can return the correct array."""
        array1 = math.astensor(np.random.random((1,5,7,8))) # where 1 is the batch.
        array2 = math.astensor(np.random.random((3,5,7,8))) # where 3 is the batch.
        fock1 = Fock(array1)
        fock2 = Fock(array2)
        fock_test = fock1[2] @ fock2[2]  # contract wires 2 on each (inner product)
        assert fock_test.array.shape == (3, 5, 7, 5, 7) # note that the batch is 1 * 3
        assert np.allclose(fock_test, np.einsum("bcde, bfge -> bcdfg",array1, array2))

    def test_trace(self):
        pass

    def test_reorder(self):
        pass
