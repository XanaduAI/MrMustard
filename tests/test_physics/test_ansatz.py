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

import numpy as np
from hypothesis import given

from mrmustard import math
from mrmustard.physics.ansatze import PolyExpAnsatz
from tests.random import complex_matrix, complex_nonzero, complex_vector


@given(A=complex_matrix(2,2), b=complex_vector(2), c=complex_nonzero)
def test_PolyExpAnsatz(A, b, c):
    """Test that the PolyExpAnsatz class is initialized correctly"""
    ansatz = PolyExpAnsatz(A, b, c)
    assert np.allclose(ansatz.mat[0], A)
    assert np.allclose(ansatz.vec[0], b)
    assert np.allclose(ansatz.array[0], c)

# test adding two PolyExpAnsatz objects
@given(A=complex_matrix(2,2), b=complex_vector(2), c=complex_nonzero,
       A2=complex_matrix(2,2), b2=complex_vector(2), c2=complex_nonzero)
def test_PolyExpAnsatz_add(A, b, c, A2, b2, c2):
    """Test that we can add two PolyExpAnsatz objects"""
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    ansatz3 = ansatz + ansatz2
    assert np.allclose(ansatz3.mat[0], A)
    assert np.allclose(ansatz3.vec[0], b)
    assert np.allclose(ansatz3.array[0], c)
    assert np.allclose(ansatz3.mat[1], A2)
    assert np.allclose(ansatz3.vec[1], b2)
    assert np.allclose(ansatz3.array[1], c2)

# test multiplying two PolyExpAnsatz objects
@given(A=complex_matrix(2,2), b=complex_vector(2), c=complex_nonzero,
       A2=complex_matrix(2,2), b2=complex_vector(2), c2=complex_nonzero)
def test_PolyExpAnsatz_mul(A, b, c, A2, b2, c2):
    """Test that we can multiply two PolyExpAnsatz objects"""
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    ansatz3 = ansatz * ansatz2
    assert np.allclose(ansatz3.mat[0], A + A2)
    assert np.allclose(ansatz3.vec[0], b + b2)
    assert np.allclose(ansatz3.array[0], c * c2)


# test multiplying a PolyExpAnsatz object by a scalar
@given(A=complex_matrix(3,3), b=complex_vector(3), c=complex_nonzero, d=complex_nonzero)
def test_PolyExpAnsatz_mul_scalar(A, b, c, d):
    """Test that we can multiply a PolyExpAnsatz object by a scalar"""
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = ansatz * d
    assert np.allclose(ansatz2.mat[0], A)
    assert np.allclose(ansatz2.vec[0], b)
    assert np.allclose(ansatz2.array[0], d * c)


# test calling the PolyExpAnsatz object
@given(A=complex_matrix(3,3), b=complex_vector(3), c=complex_nonzero, z=complex_vector(3))
def test_PolyExpAnsatz_call(A, b, c, z):
    """Test that we can call the PolyExpAnsatz object"""
    ansatz = PolyExpAnsatz(A, b, c)
    assert np.allclose(ansatz(z), c * np.exp(0.5 * z.conj().T @ A @ z + b.conj().T @ z))


# test tensor product of two PolyExpAnsatz objects
@given(A=complex_matrix(1,1), b=complex_vector(1), c=complex_nonzero,
       A2=complex_matrix(2,2), b2=complex_vector(2), c2=complex_nonzero)
def test_PolyExpAnsatz_kron(A, b, c, A2, b2, c2):
    """Test that we can tensor product two PolyExpAnsatz objects"""
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    ansatz3 = ansatz & ansatz2
    assert np.allclose(ansatz3.mat[0], math.block_diag(A, A2))
    assert np.allclose(ansatz3.vec[0], math.concat([b, b2], -1))
    assert np.allclose(ansatz3.array[0], c * c2)

# test equality
def test_PolyExpAnsatz_eq():
    """Test that we can compare two PolyExpAnsatz objects"""
    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    b = np.random.rand(3) + 1j * np.random.rand(3)
    c = np.random.rand(1)[0] + 1j * np.random.rand(1)[0]
    A2 = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    b2 = np.random.rand(3) + 1j * np.random.rand(3)
    c2 = np.random.rand(1)[0] + 1j * np.random.rand(1)[0]
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    assert ansatz == ansatz
    assert ansatz2 == ansatz2
    assert ansatz != ansatz2
    assert ansatz2 != ansatz