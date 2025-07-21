# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module contains functions for performing calculations on objects in the Gaussian representations.
"""

from __future__ import annotations

from mrmustard import math, settings
from mrmustard.utils.typing import Matrix, Scalar, Vector


def number_means(cov: Matrix, means: Vector) -> Vector:
    r"""Returns the photon number means vector given a Wigner covariance matrix and a means vector.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector

    Returns:
        Vector: the photon number means vector
    """
    N = means.shape[-1] // 2
    return (
        means[:N] ** 2
        + means[N:] ** 2
        + math.diag_part(cov[:N, :N])
        + math.diag_part(cov[N:, N:])
        - settings.HBAR
    ) / (2 * settings.HBAR)


def purity(cov: Matrix) -> Scalar:
    r"""Returns the purity of the state with the given covariance matrix.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        float: the purity
    """
    return 1 / math.sqrt(math.det((2 / settings.HBAR) * cov))


def symplectic_eigenvals(cov: Matrix) -> Vector:
    r"""Returns the sympletic eigenspectrum of a covariance matrix.

    For a pure state, we expect the sympletic eigenvalues to be 1.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        List[float]: the sympletic eigenvalues
    """
    J = math.J(cov.shape[-1] // 2)  # create a sympletic form
    M = math.matmul(1j * J, cov * (2 / settings.HBAR))
    vals = math.abs(math.eigvals(M))  # compute the eigenspectrum
    vals = math.sort(vals)
    return vals[::2]  # return the even eigenvalues


def von_neumann_entropy(cov: Matrix) -> float:
    r"""Returns the Von Neumann entropy.

    For a pure state, we expect the Von Neumann entropy to be 0.

    Reference: (https://arxiv.org/pdf/1110.3234.pdf), Equations 46-47.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        float: the Von Neumann entropy
    """

    def g(x):
        return math.xlogy((x + 1) / 2, (x + 1) / 2) - math.xlogy((x - 1) / 2, (x - 1) / 2 + 1e-9)

    symp_vals = symplectic_eigenvals(cov)
    return math.sum(g(symp_vals))


def fidelity(mu1: Vector, cov1: Matrix, mu2: Vector, cov2: Matrix) -> float:
    r"""Returns the fidelity of two gaussian states.

    Reference: `arXiv:2102.05748 <https://arxiv.org/pdf/2102.05748.pdf>`_, equations 95-99.
    Note that we compute the square of equation 98.

    Args:
        mu1 (Vector): the means vector of state 1
        mu2 (Vector): the means vector of state 2
        cov1 (Matrix): the covariance matrix of state 1
        cov1 (Matrix): the covariance matrix of state 2

    Returns:
        float: the fidelity
    """

    cov1 = math.cast(cov1 / settings.HBAR, "complex128")  # convert to units where hbar = 1
    cov2 = math.cast(cov2 / settings.HBAR, "complex128")  # convert to units where hbar = 1

    mu1 = math.cast(mu1, "complex128")
    mu2 = math.cast(mu2, "complex128")
    deltar = (mu2 - mu1) / math.sqrt(
        settings.HBAR,
        dtype=mu1.dtype,
    )  # convert to units where hbar = 1
    J = math.J(cov1.shape[0] // 2)
    I = math.eye(cov1.shape[0])
    J = math.cast(J, "complex128")
    I = math.cast(I, "complex128")

    cov12_inv = math.inv(cov1 + cov2)

    V = math.transpose(J) @ cov12_inv @ ((1 / 4) * J + cov2 @ J @ cov1)

    W = -2 * (V @ (1j * J))
    W_inv = math.inv(W)
    matsqrtm = math.sqrtm(
        I - W_inv @ W_inv,
    )  # this also handles the case where the input matrix is close to zero
    f0_top = math.det((matsqrtm + I) @ (W @ (1j * J)))
    f0_bot = math.det(cov1 + cov2)

    f0 = math.sqrt(f0_top / f0_bot)  # square of equation 98

    dot = math.sum(
        math.transpose(deltar) * math.matvec(cov12_inv, deltar),
    )  # computing (mu2-mu1)/sqrt(hbar).T @ cov12_inv @ (mu2-mu1)/sqrt(hbar)

    _fidelity = f0 * math.exp((-1 / 2) * dot)  # square of equation 95

    return math.real(_fidelity)
