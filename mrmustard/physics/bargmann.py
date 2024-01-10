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
This module contains functions for transforming to the Bargmann representation.
"""
from typing import Sequence, Tuple

import numpy as np

from mrmustard import math, settings
from mrmustard.physics.husimi import pq_to_aadag, wigner_to_husimi
from mrmustard.utils.typing import ComplexMatrix, ComplexVector


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
    den_C = math.sqrt(math.det(Q), dtype=num_C.dtype)
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
    # now A and B have order [out_r, in_r out_l, in_l].
    return A, B, math.cast(C, "complex128")


def wigner_to_bargmann_U(X, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    """
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    return A[2 * N :, 2 * N :], B[2 * N :], math.sqrt(C)


def complex_gaussian_integral(
    Abc: tuple, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...], measure: float = -1
):
    r"""Computes the Gaussian integral of the exponential of a complex quadratic form.
    The integral is defined as (note that in general we integrate over a subset of 2m dimensions):

    :math:`\int_{C^m} F(z) d\mu(z)`

    where

    :math:`F(z) = \textrm{exp}(-0.5 z^T A z + b^T z)`

    Here, ``z`` is an ``n``-dim complex vector, ``A`` is an ``n x n`` complex matrix,
    ``b`` is an ``n``-dim complex vector, ``c`` is a complex scalar, and :math:`d\mu(z)`
    is a non-holomorphic complex measure over a subset of m pairs of z,z* variables. These
    are specified by the indices ``idx_z`` and ``idx_zconj``. The ``measure`` parameter is
    the exponent of the measure:

    :math: `dmu(z) = \textrm{exp}(m * |z|^2) \frac{d^{2n}z}{\pi^n} = \frac{1}{\pi^n}\textrm{exp}(m * |z|^2) d\textrm{Re}(z) d\textrm{Im}(z)`

    Arguments:
        A,b,c: the ``(A,b,c)`` triple
        idx_z: the tuple of indices of the z variables
        idx_zconj: the tuple of indices of the z* variables
        measure: the exponent of the measure (default is -1: Bargmann measure)

    Returns:
        The ``(A,b,c)`` triple of the result of the integral
    """
    A, b, c = Abc
    if len(idx_z) != len(idx_zconj):
        raise ValueError("idx_z and idx_zconj must have the same length")
    n = len(idx_z)
    idx = tuple(idx_z) + tuple(idx_zconj)
    not_idx = tuple(i for i in range(A.shape[-1]) if i not in idx)

    I = math.eye(n, dtype=A.dtype)
    Z = math.zeros((n, n), dtype=A.dtype)
    X = math.block([[Z, I], [I, Z]])
    M = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2) + X * measure
    D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
    R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)

    bM = math.gather(b, idx, axis=-1)
    bR = math.gather(b, not_idx, axis=-1)

    A_post = R - math.matmul(D, math.inv(M), math.transpose(D))
    b_post = bR - math.matvec(D, math.solve(M, bM))
    c_post = (
        c * math.sqrt((-1) ** n / math.det(M)) * math.exp(-0.5 * math.sum(bM * math.solve(M, bM)))
    )

    return A_post, b_post, c_post


def join_Abc(Abc1, Abc2):
    r"""Joins two ``(A,b,c)`` triples into a single ``(A,b,c)`` triple by block addition of the ``A``
    matrices and concatenating the ``b`` vectors.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple

    Returns:
        The joined ``(A,b,c)`` triple
    """
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    A12 = math.block_diag(A1, A2)
    b12 = math.concat([b1, b2], axis=-1)
    c12 = math.outer(c1, c2)
    return A12, b12, c12


def reorder_abc(Abc: tuple, order: Sequence[int]):
    r"""
    Reorders the indices of the A matrix and b vector of an (A,b,c) triple.

    Arguments:
        Abc: the ``(A,b,c)`` triple
        order: the new order of the indices

    Returns:
        The reordered ``(A,b,c)`` triple
    """
    A, b, c = Abc
    A = math.gather(math.gather(A, order, axis=-1), order, axis=-2)
    b = math.gather(b, order, axis=-1)
    if len(c.shape) == len(order):
        c = math.transpose(c, order)
    return A, b, c


def contract_two_Abc(
    Abc1: Tuple[ComplexMatrix, ComplexVector, complex],
    Abc2: Tuple[ComplexMatrix, ComplexVector, complex],
    idx1: Sequence[int],
    idx2: Sequence[int],
):
    r"""
    Returns the contraction of two ``(A,b,c)`` triples.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        idx1: the indices of the first ``(A,b,c)`` triple to contract
        idx2: the indices of the second ``(A,b,c)`` triple to contract

    Returns:
        The contracted ``(A,b,c)`` triple
    """
    Abc = join_Abc(Abc1, Abc2)
    return complex_gaussian_integral(
        Abc, idx1, tuple(n + Abc1[0].shape[-1] for n in idx2), measure=-1.0
    )
