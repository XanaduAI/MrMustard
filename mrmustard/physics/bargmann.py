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
from mrmustard.physics.husimi import wigner_to_husimi, pq_to_aadag
from mrmustard import settings
from mrmustard.math import Math

math = Math()


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

    Note that here A and B are defined with inverted blocks with respect to the literature,
    otherwise the density matrix would have the left and the right indices swapped once we convert to Fock.
    By inverted blocks we mean that if A is normally defined as `A = [[A_00, A_01], [A_10, A_11]]`,
    here we define it as `A = [[A_11, A_10], [A_01, A_00]]`. For `B` we have `B = [B_0, B_1] -> B = [B_1, B_0]`.
    """
    N = cov.shape[-1] // 2
    Q, beta = wigner_to_husimi(cov, means)
    A = math.matmul(
        cayley(pq_to_aadag(cov), c=0.5), math.Xmat(N)
    )  # X on the right, so the index order will be rho_{left,right}:
    B = math.solve(Q, beta)  # no conjugate, so that the index order will be rho_{left,right}
    C = math.exp(-0.5 * math.sum(math.conj(beta) * B)) / math.sqrt(math.det(Q))
    return A, B, C


def wigner_to_bargmann_psi(cov, means):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann A,B,C triple
    for a Hilbert vector (i.e. for M modes, A has shape M x M and B has shape M).
    """
    N = cov.shape[-1] // 2
    A, B, C = wigner_to_bargmann_rho(cov, means)
    # NOTE: with A_rho and B_rho defined with inverted blocks, we now keep the first half rather than the second
    return A[:N, :N], B[:N], math.sqrt(C)


def wigner_to_bargmann_Choi(X, Y, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a channel (i.e. for M modes, A has shape 4M x 4M and B has shape 4M).
    We have freedom to choose the order of the indices of the Choi matrix by rearranging the `MxM` blocks of A and the M-subvectors of B.
    Here we choose the order `[out_l, in_l out_r, in_r]` (`in_l` and `in_r` to be contracted with the left and right indices of the density matrix)
    so that after the contraction the result has the right order `[out_l, out_r]`."""
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
    A = math.matmul(A, math.Xmat(2 * N))  # yes: X on the right
    b = math.matvec(xi_inv, d)
    B = math.matvec(math.conj(R), math.concat([b, -math.matvec(XT, b)], axis=-1)) / math.sqrt(
        settings.HBAR, dtype=R.dtype
    )
    B = math.concat([B[2 * N :], B[: 2 * N]], axis=-1)  # yes: opposite order
    C = math.exp(-0.5 * math.sum(d * b) / settings.HBAR) / math.sqrt(math.det(xi), dtype=b.dtype)
    # now A and B have order [out_l, in_l out_r, in_r].
    return A, B, C


def wigner_to_bargmann_U(X, d):
    r"""Converts the wigner representation in terms of covariance matrix and mean vector into the Bargmann `A,B,C` triple
    for a unitary (i.e. for `M` modes, `A` has shape `2M x 2M` and `B` has shape `2M`).
    """
    N = X.shape[-1] // 2
    A, B, C = wigner_to_bargmann_Choi(X, math.zeros_like(X), d)
    # NOTE: with A_Choi and B_Choi defined with inverted blocks, we now keep the first half rather than the second
    return A[: 2 * N, : 2 * N], B[: 2 * N], math.sqrt(C)
