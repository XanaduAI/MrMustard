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

"""optimization tests"""

from hypothesis import given, strategies as st

import numpy as np
import tensorflow as tf

from scipy.stats import unitary_group, ortho_group

from thewalrus.symplectic import is_symplectic, sympmat
from thewalrus.random import random_symplectic

from mrmustard.training.parameter_update import update_symplectic, update_unitary

from mrmustard.math import Math

math = Math()


def is_unitary(M, rtol=1e-05, atol=1e-08):
    """Testing if the matrix M is unitary"""
    M_dagger = np.transpose(M.conj())
    return np.allclose(M @ M_dagger, np.identity(M.shape[1]), rtol=rtol, atol=atol)


def is_orthogonal(M, rtol=1e-05, atol=1e-08):
    """Testing if the matrix M is orthogonal"""
    M_T = np.transpose(M)
    return np.allclose(M @ M_T, np.identity(M.shape[1]), rtol=rtol, atol=atol)


@given(n=st.integers(2, 4))
def test_update_symplectic(n):
    """Testing the update of symplectic matrix remains to be symplectic"""
    S = tf.Variable(random_symplectic(n), dtype=tf.complex128)
    for i in range(20):
        dS_euclidean = tf.Variable(
            np.random.random((2 * n, 2 * n)) + 1j * np.random.random((2 * n, 2 * n))
        )
        update_symplectic([[dS_euclidean, S]], 0.01)
        assert is_symplectic(S.numpy()), "training is not stay in symplectic matrix"


@given(n=st.integers(2, 4))
def test_update_unitary(n):
    """Testing the update of unitary matrix remains to be unitary"""
    U = tf.Variable(unitary_group.rvs(dim=n), dtype=tf.complex128)
    for i in range(20):
        dU_euclidean = tf.Variable(np.random.random((n, n)) + 1j * np.random.random((n, n)))
        update_unitary([[dU_euclidean, U]], 0.01)
        assert is_unitary(U.numpy()), "training is not stay in unitary matrix"
        sym = np.block(
            [[np.real(U.numpy()), -np.imag(U.numpy())], [np.imag(U.numpy()), np.real(U.numpy())]]
        )
        assert is_symplectic(sym), "training is not stay in symplectic matrix"
        assert is_orthogonal(sym), "training is not stay in orthogonal matrix"


# @given(n=st.integers(2, 4))
# def test_update_unitary_real(n):
#    """Testing the update of orthogonal matrix remains to be orthogonal"""
#    O = tf.Variable(ortho_group.rvs(dim=n), dtype=tf.complex128)
#    for i in range(20):
#        dO_euclidean = tf.Variable(np.random.random((n, n)), dtype=tf.complex128)
#        update_unitary([[dO_euclidean, O]], 0.01)
#        assert is_unitary(O.numpy()), "training is not stay in unitary matrix"
#        sym = np.block([[np.real(O.numpy()),-np.imag(O.numpy())],[np.imag(O.numpy()),np.real(O.numpy())]])
#        assert is_symplectic(sym), "training is not stay in symplectic matrix"
#        assert is_orthogonal(sym), "training is not stay in orthogonal matrix"
