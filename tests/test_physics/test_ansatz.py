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
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mrmustard import math
from mrmustard.physics.ansatze import PolyExpAnsatz
from tests.random import complex_vector

# Complex number strategy
complex_number = st.complex_numbers(min_magnitude=1e-9, max_magnitude=1, allow_infinity=False, allow_nan=False)

# Size strategy
size = st.integers(min_value=1, max_value=9)

@st.composite
def Abc_triple(draw, n = None):
    n = n or draw(size)
    
    # Complex symmetric matrix A
    A = draw(arrays(dtype=complex, shape=(n, n), elements=complex_number))
    A = 0.5 * (A + A.T)  # Make it symmetric
    
    # Complex vector b
    b = draw(arrays(dtype=complex, shape=n, elements=complex_number))
    
    # Complex scalar c
    c = draw(complex_number)
    
    return A, b, c

@given(Abc=Abc_triple())
def test_PolyExpAnsatz(Abc):
    """Test that the PolyExpAnsatz class is initialized correctly"""
    A, b, c = Abc
    ansatz = PolyExpAnsatz(A, b, c)
    assert np.allclose(ansatz.mat[0], A)
    assert np.allclose(ansatz.vec[0], b)
    assert np.allclose(ansatz.array[0], c)

@st.composite
def AbcAbc(draw):
    n = draw(size)
    Abc1 = draw(Abc_triple(n))
    Abc2 = draw(Abc_triple(n))
    return Abc1, Abc2

# test adding two PolyExpAnsatz objects
@given(Abc1_Abc2 = AbcAbc())
def test_PolyExpAnsatz_add(Abc1_Abc2):
    """Test that we can add two PolyExpAnsatz objects"""
    Abc1, Abc2 = Abc1_Abc2
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

# test multiplying two PolyExpAnsatz objects
@given(Abc1_Abc2 = AbcAbc())
def test_PolyExpAnsatz_mul(Abc1_Abc2):
    """Test that we can multiply two PolyExpAnsatz objects"""
    Abc1, Abc2 = Abc1_Abc2
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    ansatz = PolyExpAnsatz(A1, b1, c1)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    ansatz3 = ansatz * ansatz2
    assert np.allclose(ansatz3.mat[0], A1 + A2)
    assert np.allclose(ansatz3.vec[0], b1 + b2)
    assert np.allclose(ansatz3.array[0], c1 * c2)


# test multiplying a PolyExpAnsatz object by a scalar
@given(Abc = Abc_triple(), d=complex_number)
def test_PolyExpAnsatz_mul_scalar(Abc, d):
    """Test that we can multiply a PolyExpAnsatz object by a scalar"""
    A, b, c = Abc
    ansatz = PolyExpAnsatz(A, b, c)
    ansatz2 = ansatz * d
    assert np.allclose(ansatz2.mat[0], A)
    assert np.allclose(ansatz2.vec[0], b)
    assert np.allclose(ansatz2.array[0], d * c)

# test calling the PolyExpAnsatz object
@given(Abc = Abc_triple(), z=complex_vector())
def test_PolyExpAnsatz_call(Abc, z):
    """Test that we can call the PolyExpAnsatz object"""
    A, b, c = Abc
    assume(len(z) == A.shape[-1])
    ansatz = PolyExpAnsatz(A, b, c)
    assert np.allclose(ansatz(z*0), c)
    assert np.allclose(ansatz(z), c * np.exp(0.5 * z.conj().T @ A @ z + b.conj().T @ z))


# test tensor product of two PolyExpAnsatz objects
@given(Abc1_Abc2 = AbcAbc())
def test_PolyExpAnsatz_kron(Abc1_Abc2):
    """Test that we can tensor product two PolyExpAnsatz objects"""
    Abc1, Abc2 = Abc1_Abc2
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    ansatz = PolyExpAnsatz(A1, b1, c1)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    ansatz3 = ansatz & ansatz2
    assert np.allclose(ansatz3.mat[0], math.block_diag(A1, A2))
    assert np.allclose(ansatz3.vec[0], math.concat([b1, b2], -1))
    assert np.allclose(ansatz3.array[0], c1* c2)

# test equality
@given(Abc1_Abc2 = AbcAbc())
def test_PolyExpAnsatz_eq(Abc1_Abc2):
    """Test that we can compare two PolyExpAnsatz objects"""
    Abc1, Abc2 = Abc1_Abc2
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    ansatz = PolyExpAnsatz(A1, b1, c1)
    ansatz2 = PolyExpAnsatz(A2, b2, c2)
    assert ansatz == ansatz
    assert ansatz2 == ansatz2
    assert ansatz != ansatz2
    assert ansatz2 != ansatz