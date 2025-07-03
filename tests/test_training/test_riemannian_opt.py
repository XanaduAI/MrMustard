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

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import unitary_group
from thewalrus.random import random_symplectic
from thewalrus.symplectic import is_symplectic

from mrmustard import math, settings
from mrmustard.math.parameters import update_orthogonal, update_symplectic, update_unitary


@pytest.mark.requires_backend("tensorflow")
def is_unitary(M, rtol=1e-05, atol=1e-08):
    """Testing if the matrix M is unitary"""
    M_dagger = np.transpose(M.conj())
    return math.allclose(M @ M_dagger, np.identity(M.shape[-1]), rtol=rtol, atol=atol)


@pytest.mark.requires_backend("tensorflow")
def is_orthogonal(M, rtol=1e-05, atol=1e-08):
    """Testing if the matrix M is orthogonal"""
    M_T = np.transpose(M)
    return math.allclose(M @ M_T, np.identity(M.shape[-1]), rtol=rtol, atol=atol)


@pytest.mark.requires_backend("tensorflow")
@given(n=st.integers(2, 4))
def test_update_symplectic(n):
    """Testing the update of symplectic matrix remains to be symplectic"""
    rng = settings.rng
    S = math.new_variable(random_symplectic(n), name=None, dtype="complex128", bounds=None)
    for _ in range(20):
        dS_euclidean = math.new_variable(
            rng.random((2 * n, 2 * n)) + 1j * rng.random((2 * n, 2 * n)),
            name=None,
            dtype="complex128",
            bounds=None,
        )
        update_symplectic([[dS_euclidean, S]], 0.01)
        assert is_symplectic(math.asnumpy(S)), (
            "training step does not result in a symplectic matrix"
        )


@pytest.mark.requires_backend("tensorflow")
@given(n=st.integers(2, 4))
def test_update_unitary(n):
    """Testing the update of unitary matrix remains to be unitary"""
    rng = settings.rng
    U = math.new_variable(unitary_group.rvs(dim=n), name=None, dtype="complex128", bounds=None)
    for _ in range(20):
        dU_euclidean = rng.random((n, n)) + 1j * rng.random((n, n))
        update_unitary([[dU_euclidean, U]], 0.01)
        assert is_unitary(math.asnumpy(U)), "training step does not result in a unitary matrix"
        sym = np.block(
            [
                [np.real(math.asnumpy(U)), -np.imag(math.asnumpy(U))],
                [np.imag(math.asnumpy(U)), np.real(math.asnumpy(U))],
            ],
        )
        assert is_symplectic(sym), "training step does not result in a symplectic matrix"
        assert is_orthogonal(sym), "training step does not result in an orthogonal matrix"


@pytest.mark.requires_backend("tensorflow")
@given(n=st.integers(2, 4))
def test_update_orthogonal(n):
    """Testing the update of orthogonal matrix remains to be orthogonal"""
    rng = settings.rng
    O = math.new_variable(math.random_orthogonal(n), name=None, dtype="complex128", bounds=None)
    for _ in range(20):
        dO_euclidean = rng.random((n, n)) + 1j * rng.random((n, n))
        update_orthogonal([[dO_euclidean, O]], 0.01)
        assert is_unitary(math.asnumpy(O)), "training step does not result in a unitary matrix"
        ortho = np.block(
            [
                [np.real(math.asnumpy(O)), -math.zeros_like(math.asnumpy(O))],
                [math.zeros_like(math.asnumpy(O)), np.real(math.asnumpy(O))],
            ],
        )
        assert is_symplectic(ortho), "training step does not result in a symplectic matrix"
        assert is_orthogonal(ortho), "training step does not result in an orthogonal matrix"
