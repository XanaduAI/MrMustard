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


# Recurrencies for Fock-Bargmann amplitudes put together a strategy for
# enumerating the indices in a specific order and functions for calculating
# which neighbours to use in the calculation of the amplitude at a given index.
# In summary, they return the value of the amplitude at the given index by following
# a recipe made of two parts. The function to recompute A and b is determined by
# which neighbours are used.

"""Fock-Bargmann recurrence relation steps optimized for beamsplitter."""

import numpy as np
from numba import njit

from mrmustard.math.lattice import steps
from mrmustard.typing import ComplexMatrix, ComplexTensor, ComplexVector

SQRT = np.sqrt(np.arange(100000))

__all__ = ["beamsplitter", "beamsplitter_vjp", "beamsplitter_schwinger"]


@njit
def beamsplitter(
    shape: tuple[int, int, int, int], theta: float, phi: float, dtype=np.complex128
) -> ComplexTensor:  # pragma: no cover
    r"""Calculates the Fock representation of the beamsplitter.
    It takes advantage of input-output particle conservation (m+n=p+q)
    to avoid one for loop. Inspired from the original implementation in
    the walrus by @ziofil. Here is how the parameters are used in the
    code (see eq. 73-75 in https://arxiv.org/abs/2004.11002):

    A = [[0,V],[v^T,0]]   # BS bargmann matrix
    V = [[ct, -st expm],  # BS unitary
         [st exp, ct]]

    Args:
        shape (tuple[int, int, int, int]): shape of the Fock representation
        theta (float): beamsplitter angle
        phi (float): beamsplitter phase
        dtype (np.dtype): data type of the Fock representation

    Returns:
        array (ComplexTensor): The Fock representation of the gate
    """
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    stc = np.conj(st)

    M, N, P, Q = shape
    G = np.zeros(shape, dtype=dtype)
    G[0, 0, 0, 0] = 1.0

    # rank 3
    for m in range(M):
        for n in range(N - m):
            p = m + n
            if 0 < p < P:
                G[m, n, p, 0] = (
                    ct * SQRT[m] / SQRT[p] * G[m - 1, n, p - 1, 0]
                    + st * SQRT[n] / SQRT[p] * G[m, n - 1, p - 1, 0]
                )

    # rank 4
    for m in range(M):
        for n in range(N):
            for p in range(P):
                q = m + n - p
                if 0 < q < Q:
                    G[m, n, p, q] = (
                        -stc * SQRT[m] / SQRT[q] * G[m - 1, n, p, q - 1]
                        + ct * SQRT[n] / SQRT[q] * G[m, n - 1, p, q - 1]
                    )
    return G


@njit
def beamsplitter_vjp(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    theta: float,
    phi: float,
) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
    r"""Beamsplitter gradients with respect to theta and phi.
    This function could return dL/dA, dL/db, dL/dc like its vanilla counterpart,
    but it is more efficient to include this chain rule step in the numba function,
    since we can.

    We use these derivatives of the BS unitary:

    dVdt = [[-st, -ct exp],
            [ct exp, -st]]
    dVdphi = [[0, -i ct expm],
              [i ct exp, 0]]

    Args:
        G (np.ndarray): Tensor result of the forward pass
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor
        theta (float): beamsplitter angle
        phi (float): beamsplitter phase

    Returns:
        tuple[float, float]: dL/dtheta, dL/dphi
    """
    M, N, P, Q = G.shape

    # init gradients
    dA = np.zeros((4, 4), dtype=np.complex128)
    db = np.zeros(4, dtype=np.complex128)
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # rank 3
    for m in range(M):
        for n in range(N - m):
            p = m + n
            if 0 < p < P:
                dA, db = steps.vanilla_step_grad(G, (m, n, p, 0), dA, db)
                dLdA += dA * dLdG[m, n, p, 0]
                dLdb += db * dLdG[m, n, p, 0]

    # rank 4
    for m in range(M):
        for n in range(N):
            for p in range(P):
                q = m + n - p
                if 0 < q < Q:
                    dA, db = steps.vanilla_step_grad(G, (m, n, p, q), dA, db)
                    dLdA += dA * dLdG[m, n, p, q]
                    dLdb += db * dLdG[m, n, p, q]

    st = np.sin(theta)
    ct = np.cos(theta)
    e = np.exp(1j * phi)
    em = np.exp(-1j * phi)

    # omitting bottom-left block because dLdA should be zero there
    dLdtheta = 2 * np.real(
        -st * dLdA[0, 2] - ct * em * dLdA[0, 3] + ct * e * dLdA[1, 2] - st * dLdA[1, 3]
    )
    dLdphi = 2 * np.real(1j * st * em * dLdA[0, 3] + 1j * st * e * dLdA[1, 2])

    return dLdtheta, dLdphi


def beamsplitter_schwinger(shape, theta, phi, max_N=None):
    r"""Returns the Fock representation of the beamsplitter up to
    the given cutoff for each of the two modes.

    This implementation is in pure python (so it's slower than the numba version),
    but it's numerically stable up to arbitrary cutoffs.

    In this implementation we split the two-mode Fock basis into finite subsets spanned
    by |m,n> with m+n=const, i.e. {|0,0>}, {|1,0>,|0,1>}, {|2,0>, |1,1>, |0,2>}, etc...
    A beamsplitter acts unitariliy in each of these subspaces without mixing them with each other,
    i.e. in this basis the beamsplitter would be a block-diagonal matrix.
    This means we can construct the BS matrix by first calculating the BS unitaries
    in each of these subspaces.
    This can be done using the matrix exponential, which is numerically stable.
    We couldn't do this in the original basis because it was infinite-dimensional
    and we had to truncate it at some point.

    Arguments:
        shape (int, int, int, int): The shape of the output tensor. Only shapes of the form (i,k,i,k) are supported.
        theta (float): The angle of the beamsplitter.
        phi (float): The phase of the beamsplitter.
        max_N (int): The maximum total photon number to include in the calculation.

    Returns:
        np.ndarray: The beamsplitter in the Fock basis.
    """
    c1, c2, c3, c4 = shape
    if c1 != c3 or c2 != c4:
        raise ValueError("The Schwinger method only supports shapes of the form (i,k,i,k).")
    # create output tensor
    U = np.zeros(shape, dtype="complex128")

    # loop over subspaces of constant photon number N up to max_N
    if max_N is None or max_N > c1 + c2 - 2:
        max_N = c1 + c2 - 2
    for N in range(max_N + 1):
        # construct the N+1 x N+1 unitary for this subspace
        diag = np.exp(1j * phi) * np.sqrt(np.arange(N, 0, -1) * np.arange(1, N + 1, 1))
        iJy = np.diag(diag, k=-1) - np.diag(np.conj(diag), k=1)
        E, V = np.linalg.eig(theta * iJy)
        block = V @ np.diag(np.exp(E)) @ np.conj(V.T)
        # insert the elements of the block into the output tensor
        for i in range(max(0, N + 1 - c1), min(N + 1, c1)):
            for j in range(max(0, N + 1 - c1), min(N + 1, c2)):
                U[N - i, i, N - j, j] = block[i, j]
    return U
