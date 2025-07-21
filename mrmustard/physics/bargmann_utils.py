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
from mrmustard.utils.typing import ComplexMatrix, Matrix, Scalar, Vector


def bargmann_Abc_to_phasespace_cov_means(
    A: Matrix,
    b: Vector,
    c: Scalar,
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    Function to derive the covariance matrix and mean vector of a Gaussian state from its Wigner characteristic function in ABC form.

    The covariance matrix and mean vector can be used to write the characteristic function of a Gaussian state

    :math:
        \Chi_G(r) = \exp\left( -\frac{1}{2}r^T \Omega^T cov \Omega r + i r^T\Omega^T mean \right),

    and the Wigner function of a Gaussian state:

    :math:
        W_G(r) = \frac{1}{\sqrt{\Det(cov)}} \exp\left( -\frac{1}{2}(r - mean)^T cov^{-1} (r-mean) \right).

    The internal expression of our Gaussian state :math:`\rho` is in Bargmann representation, one can write the characteristic function of a Gaussian state in Bargmann representation as

    :math:
        \Chi_G(\alpha) = \Tr(\rho D) = c \exp\left( -\frac{1}{2}\alpha^T A \alpha + \alpha^T b \right).

    This function is to go from the Abc triple in characteristic phase space into the covariance and mean vector for Gaussian state.

    Args:
        A, b, c: The ``(A, b, c)`` triple of the state in characteristic phase space.

    Returns:
        The covariance matrix, mean vector and coefficient of the state in phase space.
    """
    num_modes = A.shape[-1] // 2
    Omega = math.cast(math.transpose(math.J(num_modes)), dtype=math.complex128)
    W = math.transpose(math.conj(math.rotmat(num_modes)))
    cov = -Omega @ W @ A @ math.transpose(W) @ math.transpose(Omega) * settings.HBAR
    mean = 1j * math.matvec(Omega @ W, b) * math.sqrt(settings.HBAR, dtype=math.complex128)
    return cov, mean, c


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


def au2Symplectic(A):
    r"""
    helper for finding the Au of a unitary from its symplectic rep.
    Au : in bra-ket order
    """
    # A represents the A matrix corresponding to unitary U
    A = A * (1.0 + 0.0 * 1j)
    m = A.shape[-1] // 2
    # identifying blocks of A_u
    u_2 = A[..., :m, m:]
    u_3 = A[..., m:, m:]

    transposed_u_2 = math.einsum("...ij->...ji", u_2)
    # The formula to apply comes here
    S_1 = math.conj(math.inv(transposed_u_2))
    S_2 = -math.conj(math.solve(transposed_u_2, u_3))
    S_3 = math.conj(S_2)
    S_4 = math.conj(S_1)

    S = math.block([[S_1, S_2], [S_3, S_4]])

    transformation = (
        1
        / np.sqrt(2)
        * math.block(
            [
                [
                    math.eye(m, dtype=math.complex128),
                    math.eye(m, dtype=math.complex128),
                ],
                [
                    -1j * math.eye(m, dtype=math.complex128),
                    1j * math.eye(m, dtype=math.complex128),
                ],
            ],
        )
    )

    return math.real(transformation @ S @ math.conj(math.transpose(transformation)))


def symplectic2Au(S):
    r"""
    The inverse of au2Symplectic i.e., returns symplectic, given Au

    S: symplectic in XXPP order
    """
    m = S.shape[-1] // 2
    # the following lines of code transform the quadrature symplectic matrix to
    # the annihilation one
    R = math.rotmat(m)
    S = R @ S @ math.dagger(R)
    # identifying blocks of S
    S_1 = S[..., :m, :m]
    S_2 = S[..., :m, m:]

    S_1_transposed = math.einsum("...ij->...ji", S_1)

    # the formula to apply comes here
    A_1 = S_2 @ math.conj(math.inv(S_1))  # use solve for inverse
    A_2 = math.conj(math.inv(S_1_transposed))
    A_3 = math.einsum("...ij->...ji", A_2)
    A_4 = -math.conj(math.solve(S_1, S_2))

    return math.block([[A_1, A_2], [A_3, A_4]])


def XY_of_channel(A: ComplexMatrix):
    r"""
    Outputting the X and Y matrices corresponding to a channel determined by the "A"
    matrix.

    Args:
        A: the A matrix of the channel
    """
    n = A.shape[-1] // 2
    m = n // 2

    # here we transform to the other convention for wires i.e. {out-bra, out-ket, in-bra, in-ket}
    A_out = math.block(
        [
            [A[..., :m, :m], A[..., :m, 2 * m : 3 * m]],
            [A[..., 2 * m : 3 * m, :m], A[..., 2 * m : 3 * m, 2 * m : 3 * m]],
        ],
    )
    R = math.block(
        [
            [A[..., :m, m : 2 * m], A[..., :m, 3 * m :]],
            [A[..., 2 * m : 3 * m, m : 2 * m], A[..., 2 * m : 3 * m, 3 * m :]],
        ],
    )
    X_tilde = (
        -math.inv(math.eye(n, dtype=math.complex128) - math.Xmat(m) @ A_out)
        @ math.Xmat(m)
        @ R
        @ math.Xmat(m)
    )
    transformation = math.block(
        [
            [math.eye(m, dtype=math.complex128), math.eye(m, dtype=math.complex128)],
            [
                -1j * math.eye(m, dtype=math.complex128),
                1j * math.eye(m, dtype=math.complex128),
            ],
        ],
    )
    X = -transformation @ X_tilde @ math.conj(transformation).T / 2

    sigma_H = math.inv(math.eye(n) - math.Xmat(m) @ A_out)  # the complex-Husimi covariance matrix

    N = sigma_H[..., m:, m:]
    M = sigma_H[..., :m, m:]
    sigma = (
        math.block([[math.real(N + M), math.imag(N + M)], [math.imag(M - N), math.real(N - M)]])
        - math.eye(n) / 2
    )
    X_transposed = math.einsum("...ij->...ji", X)
    Y = sigma - X @ X_transposed / 2
    math.error_if(
        X,
        math.norm(math.imag(X)) > settings.ATOL,
        "Invalid input for the A matrix of channel, caused by an imaginary X matrix.",
    )
    math.error_if(
        Y,
        math.norm(math.imag(Y)) > settings.ATOL,
        "Invalid input for the A matrix of channel, caused by an imaginary Y matrix.",
    )
    return math.real(X), math.real(Y) * settings.HBAR
