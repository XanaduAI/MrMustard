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

"""
This module contains functions for performing calculations on objects in the Bargmann representations.
"""
import numpy as np

from mrmustard import math, settings
from mrmustard.physics.husimi import pq_to_aadag, wigner_to_husimi


def cayley(X, c):
    r"""Returns the Cayley transform of a matrix:
    :math:`cay(X) = (X - cI)(X + cI)^{-1}`

    Args:
        c (float): the parameter of the Cayley transform
        X (Tensor): a matrix

    Returns:
        Tensor: the Cayley transform of X
    """
    I = math.eye(X.shape[0], dtype=X.dtype)
    return math.solve(X + c * I, X - c * I)


def wigner_to_bargmann_rho(cov, means):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a density matrix (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    The order of the rows/columns of A and B corresponds to a density matrix with the usual ordering of the indices.

    Note that here A and B are defined with respect to the literature.
    """
    N = cov.shape[-1] // 2
    A = math.matmul(math.Xmat(N), cayley(pq_to_aadag(cov), c=0.5))
    Q, beta = wigner_to_husimi(cov, means)
    b = math.solve(Q, beta)
    B = math.conj(b)
    num_C = math.exp(-0.5 * math.sum(math.conj(beta) * b))
    detQ = math.det(Q)
    den_C = math.sqrt(detQ, dtype=num_C.dtype)
    C = num_C / den_C
    return A, B, C


def wigner_to_bargmann_psi(cov, means):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann A,B,C triple
    for a Hilbert vector (i.e. for M modes, A has shape M x M and B has shape M).
    """
    N = cov.shape[-1] // 2
    A, B, C = wigner_to_bargmann_rho(cov, means)
    return A[N:, N:], B[N:], math.sqrt(C)
    # NOTE: c for th psi is to calculated from the global phase formula.


def wigner_to_bargmann_Choi(X, Y, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a channel (i.e. for M modes, A has shape 4M x 4M and B has shape 4M)."""
    N = X.shape[-1] // 2
    I2 = math.eye(2 * N, dtype=X.dtype)
    XT = math.transpose(X)
    xi = 0.5 * (I2 + math.matmul(X, XT) + 2 * Y / settings.HBAR)
    detxi = math.det(xi)
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
    C = math.exp(-0.5 * math.sum(d * b) / settings.HBAR) / math.sqrt(detxi, dtype=b.dtype)
    # now A and B have order [out_r, in_r out_l, in_l].
    return A, B, math.cast(C, "complex128")


def wigner_to_bargmann_U(X, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    """
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    return A[2 * N :, 2 * N :], B[2 * N :], math.sqrt(C)


def Au2Symplectic(A):
    r"""
    helper for finding the Au of a unitary from its symplectic rep.
    Au : in bra-ket order
    """
    # A represents the A matrix corresponding to unitary U
    m = A.shape[-1]
    m = m // 2

    # identifying blocks of A_u
    u_2 = A[..., :m, m:]
    u_3 = A[..., m:, m:]

    # The formula to apply comes here
    S_1 = math.conj(math.inv(u_2.T))
    S_2 = -S_1 @ math.conj(u_3)
    S_3 = math.conj(S_2)
    S_4 = math.conj(S_1)

    S = math.block([[S_1, S_2], [S_3, S_4]])

    Transformation = (
        1
        / math.sqrt(2)
        * math.block([[math.eye(m), math.eye(m)], [-1j * math.eye(m), 1j * math.eye(m)]])
    )

    return math.real(Transformation @ S @ math.conj(Transformation.T))


def Symplectic2Au(S):
    r"""
    The inverse of Au2Symplectic i.e., returns symplectic, given Au

    S: symplectic in XXPP order
    """
    m = S.shape[-1]
    m = m // 2
    # the following lines of code transform the quadrature symplectic matrix to
    # the annihilation one
    Transformation = (
        1
        / math.sqrt(2)
        * math.block([[math.eye(m), math.eye(m)], [-1j * math.eye(m), 1j * math.eye(m)]])
    )
    S = np.conjugate(Transformation.T) @ S @ Transformation
    # identifying blocks of S
    S_1 = S[:m, :m]
    S_2 = S[:m, m:]

    # TODO: broadcasting/batch stuff consider a batch dimension

    # the formula to apply comes here
    A_1 = S_2 @ np.conjugate(np.linalg.pinv(S_1))  # use solve for inverse
    A_2 = np.conjugate(np.linalg.pinv(S_1.T))
    A_3 = A_2.T
    A_4 = -np.conjugate(np.linalg.pinv(S_1)) @ np.conjugate(S_2)

    A = math.block([[A_1, A_2], [A_3, A_4]])

    return A
