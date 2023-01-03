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

# pylint: disable=redefined-outer-name

"""
This module contains functions for transforming to the Bargmann representation.
"""
import numpy as np
from .husimi import pq_to_aadag
from mrmustard import settings
from mrmustard.math import Math
math = Math()


def cayley(X, c):
    r"""Returns the Cayley transform of a matrix:
    cay(X) = (X - cI)(X + cI)^{-1}

    Args:
        c (float): the parameter of the Cayley transform
        X (Tensor): a matrix

    Returns:
        Tensor: the Cayley transform of X
    """
    I = math.eye(X.shape[0], dtype=X.dtype)
    return math.matmul(X - c * I, math.inv(X + c * I))  # or solve(c+x, c-x)?


def wigner_to_bargmann_rho(cov, means):
    r"""Returns the Bargmann A,B,C triple for a density matrix (full-size).
    The order of the indices is as follows: rho_mn
    """
    N = cov.shape[-1] // 2
    sigma = pq_to_aadag(cov)
    beta = pq_to_aadag(means)
    I = math.eye(2 * N, dtype=sigma.dtype)
    Q_inv = math.inv(sigma + 0.5 * I)
    A = math.matmul(math.Xmat(N), cayley(sigma, c=0.5))
    B = math.conj(math.matvec(Q_inv, beta))
    numerator = math.exp(-0.5 * math.sum(math.conj(beta) * math.matvec(Q_inv, beta)))
    denominator = math.sqrt(math.det(sigma + 0.5 * I))
    C = numerator / denominator
    return A, B, C


def wigner_to_bargmann_psi(cov, means):
    r"""Returns the Bargmann A,B,C triple for a pure state (half-size)."""
    N = cov.shape[-1] // 2
    A, B, C = wigner_to_bargmann_rho(cov, means)
    return A[N:, N:], B[N:], math.sqrt(C)


def wigner_to_bargmann_Choi(X, Y, d):
    r"""Returns the Bargmann A,B,C triple for a channel."""
    N = X.shape[-1] // 2
    I2 = math.eye(2 * N, dtype=X.dtype)
    XT = math.transpose(X)
    xi = 0.5 * (I2 + math.matmul(X, XT) + 2 * Y / settings.HBAR)
    xi_inv = math.inv(xi)
    A = math.block(
        [
            [I2 - xi_inv, math.matmul(xi_inv, X)],
            [math.matmul(XT, xi_inv), I2 - math.matmul(math.matmul(XT, xi_inv), X)],
        ]
    )
    I = math.eye(N, dtype="complex128")
    o = math.zeros_like(I)
    R = math.block(
        [[I, 1j * I, o, o], [o, o, I, -1j * I], [I, -1j * I, o, o], [o, o, I, 1j * I]]
    ) / np.sqrt(2)
    A = math.matmul(math.matmul(R, A), math.dagger(R))
    A = math.matmul(math.Xmat(2 * N), A)
    b = math.matvec(xi_inv, d)
    B = math.matvec(math.conj(R), math.concat([b, -math.matvec(XT, b)], axis=-1)) / math.sqrt(
        settings.HBAR, dtype=R.dtype
    )
    C = math.exp(-0.5 * math.sum(d * b) / settings.HBAR) / math.sqrt(math.det(xi), dtype=b.dtype)
    return A, B, C


def wigner_to_bargmann_U(X, d):
    r"""Returns the Bargmann A,B,C triple for a unitary transformation."""
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    return A[2 * N :, 2 * N :], B[2 * N :], math.sqrt(C)
