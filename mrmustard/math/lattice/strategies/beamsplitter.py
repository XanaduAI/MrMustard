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
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

SQRT = np.sqrt(np.arange(100000))

__all__ = ["beamsplitter", "beamsplitter_schwinger", "beamsplitter_vjp", "stable_beamsplitter"]


@njit(cache=True)
def beamsplitter(
    shape: tuple[int, int, int, int],
    theta: float,
    phi: float,
    dtype=np.complex128,
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


@njit(cache=True)
def stable_beamsplitter(shape, theta, phi):  # pragma: no cover  # noqa: C901
    r"""
    Stable implementation of the Fock representation of the beamsplitter.
    It is numerically stable up to arbitrary cutoffs.
    The shape order is (out_0, out_1, in_0, in_1), assuming it acts on modes 0 and 1.

    Args:
        shape (tuple[int, int, int, int]): shape of the Fock representation
        theta (float): beamsplitter angle
        phi (float): beamsplitter phase

    Returns:
        array (ComplexTensor): The Fock representation of the gate
    """
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    stc = np.conj(st)

    M, N, P, Q = shape
    G = np.zeros(shape, dtype=np.complex128)
    G[0, 0, 0, 0] = 1.0 + 0.0j

    # rank 3
    for m in range(M):
        for n in range(min(N, P - m)):
            p = m + n
            val = 0
            pivots = 0
            if m > 0:  # pivot at (m-1, n, p, 0)
                val += ct * SQRT[p] / SQRT[m] * G[m - 1, n, p - 1, 0]
                pivots += 1
            if n > 0:  # pivot at (m, n-1, p, 0)
                val += st * SQRT[p] / SQRT[n] * G[m, n - 1, p - 1, 0]
                pivots += 1
            if p > 0:  # pivot at (m, n, p-1, 0)
                val += (
                    ct * SQRT[m] / SQRT[p] * G[m - 1, n, p - 1, 0]
                    + st * SQRT[n] / SQRT[p] * G[m, n - 1, p - 1, 0]
                )
                pivots += 1
            if m > 0 or n > 0 or p > 0:
                G[m, n, p, 0] = val / pivots

    # rank 4
    for m in range(M):
        for n in range(N):
            for p in range(max(0, m + n - Q), min(P, m + n)):
                q = m + n - p
                if 0 < q < Q:
                    val = 0
                    pivots = 0
                    if m > 0:
                        val += (
                            ct * SQRT[p] / SQRT[m] * G[m - 1, n, p - 1, q]
                            - stc * SQRT[q] / SQRT[m] * G[m - 1, n, p, q - 1]
                        )
                        pivots += 1
                    if n > 0:
                        val += (
                            st * SQRT[p] / SQRT[n] * G[m, n - 1, p - 1, q]
                            + ct * SQRT[q] / SQRT[n] * G[m, n - 1, p, q - 1]
                        )
                        pivots += 1
                    if p > 0:
                        val += (
                            ct * SQRT[m] / SQRT[p] * G[m - 1, n, p - 1, q]
                            + st * SQRT[n] / SQRT[p] * G[m, n - 1, p - 1, q]
                        )
                        pivots += 1
                    if q > 0:
                        val += (
                            -stc * SQRT[m] / SQRT[q] * G[m - 1, n, p, q - 1]
                            + ct * SQRT[n] / SQRT[q] * G[m, n - 1, p, q - 1]
                        )
                        pivots += 1
                    if m > 0 or n > 0 or p > 0 or q > 0:
                        G[m, n, p, q] = val / pivots
    return G


@njit(cache=True)
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
        for n in range(min(N, P - m)):
            p = m + n
            if 0 < p < P:
                dA, db = steps.vanilla_step_grad(G, (m, n, p, 0), dA, db)
                dLdA += dA * dLdG[m, n, p, 0]
                dLdb += db * dLdG[m, n, p, 0]

    # rank 4
    for m in range(M):
        for n in range(N):
            for p in range(max(0, m + n - Q), min(P, m + n)):
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
        -st * dLdA[0, 2] - ct * em * dLdA[0, 3] + ct * e * dLdA[1, 2] - st * dLdA[1, 3],
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


def sector_idx(N: int, shape: tuple):
    """The action of a BSgate breaks down into N-dim unitaries acting
    on each total photon number subspace, because the BS commutes with the
    total photon number operator.

    e.g. we have a two-mode ket of initial shape (4,4) (i.e. 0 to 3 photons in each mode)
    The 3x3 BS unitary on the 2-photon subspace acts on indices [2,5,8] of
    the flattened ket array.
    Here is the ordered basis vectors |m,n> by total photons m+n for the example:

    flat index  |m,n>
    0            0,0  |-- 0 photons

    1            1,0  |-- 1 photons
    4            0,1  |

    2            2,0  |-- 2 photons  <-- 2-photon subspace at indices [2,5,8]
    5            1,1  |
    8            0,2  |

    3            3,0  |-- 3 photons
    6            2,1  |
    9            1,2  |
    12           0,3  |

    7            3,1  |-- 4 photons
    10           2,2  |
    13           1,3  |

    11           3,2  |-- 5 photons
    14           2,3  |

    15           3,3  |-- 6 photons

    The left column is the flattened order. This function returns the indices in
    left column for the N-th block. E.g. sector_idx(3, (4,4)) is [3,6,9,12].

    Args:
        N (int): The total photon number of the subspace.
        shape (tuple): The shape of the array the BS is operating on.

    Returns:
        list: The flattened indices of the N-photon subspace on which the BS acts.
    """
    return [
        np.ravel_multi_index((i, N - i), shape) for i in range(N + 1) if max(i, N - i) < max(shape)
    ]


def sector_u(N: int, theta: float, phi: float) -> np.ndarray:
    """Unitary of the BSgate acting on the (N+1)-dimensional N-photon subspace.
    Each subspace is an irrep of SU(2) (Schwinger representation).

    Args:
        N (int): The total photon number of the subspace.
        theta (float): The angle of the beamsplitter.
        phi (float): The phase of the beamsplitter.

    Returns:
        np.ndarray: The ``N x N`` unitary of the BSgate acting on the N-photon subspace.
    """
    diag = np.exp(1j * phi) * np.sqrt(np.arange(N, 0, -1) * np.arange(1, N + 1, 1))
    iJy = np.diag(diag, k=-1) - np.diag(np.conj(diag), k=1)  # we want exp(i theta J_y)
    E, V = np.linalg.eigh(-1j * theta * iJy)
    return V @ np.diag(np.exp(1j * E)) @ np.conj(V.T)


def apply_BS_schwinger(theta, phi, i, j, array) -> np.ndarray:
    """Applies the BS with given theta, phi to indices i,j of the given array.

    Args:
        theta (float): The angle of the beamsplitter.
        phi (float): The phase of the beamsplitter.
        i (int): The first index of the array where the BS is applied.
        j (int): The second index of the array where the BS is applied.
        array (np.ndarray): The array to which the BS is applied.
    """
    # step 1: reshape the pair of indices to which the BS is attached
    order = [k for k in range(array.ndim) if k not in [i, j]] + [i, j]
    array = array.transpose(order)  # move the indices to the end
    shape_rest, shape = array.shape[:-2], array.shape[-2:]
    array = array.reshape((*shape_rest, -1))  # flatten the last two dimensions
    # step 2: apply each unitary to the corresponding indices
    for N in range(sum(shape) - 1):
        flat_idx = sector_idx(N, shape)
        u = sector_u(N, theta, phi)
        subset = [k for k in range(N + 1) if k < shape[0] and N - k < shape[1]]
        array[..., flat_idx] @= u[subset, ...][..., subset] if 0 < len(subset) < N else u
    # step 3: reshape back and reorder
    array = array.reshape(shape_rest + shape)
    return array.transpose(np.argsort(order))
