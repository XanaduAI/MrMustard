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

"""Gradient and VJP strategies for vanilla Fock representation calculation."""

import numpy as np
from numba import njit, prange

from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

from .core import SQRT


@njit
def vanilla_vjp(G, c, dLdG) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
    r"""Vanilla vjp function. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    # numba doesn't like tuples
    shape_arr = np.array(G.shape)

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(len(shape_arr) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # linearize G
    G_lin = G.flatten()

    # init gradients
    D = len(shape_arr)
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize the n-dim index
    flat_index = 0
    nd_index = np.ndindex(G.shape)
    next(nd_index)

    # iterate over the indices (no need to split the loop in two parts)
    for index_u in nd_index:
        flat_index += 1

        # contributions from lower neighbours
        for i in range(D):
            pivot = flat_index - strides[i]
            db[i] = SQRT[index_u[i]] * G_lin[pivot]
            dA[i, i] = 0.5 * SQRT[index_u[i]] * SQRT[index_u[i] - 1] * G_lin[pivot - strides[i]]
            for j in range(i + 1, D):
                dA[i, j] = SQRT[index_u[i]] * SQRT[index_u[j]] * G_lin[pivot - strides[j]]

        dLdA += dA * dLdG[index_u]
        dLdb += db * dLdG[index_u]

    dLdc = np.sum(G * dLdG) / c

    return dLdA, dLdb, dLdc


# TODO: thoroughly check this
@njit
def vanilla_stable_vjp(G, A, b, c, dLdG):  # pragma: no cover
    r"""Calculates the vector-Jacobian product (VJP) for the Fock representation G
    obtained with the vanilla_stable strategy with respect to the parameters A, b, c.
    Given the gradient of the loss ``dLdG`` with respect to the Fock representation ``G``,
    this function computes:

    dLdA = dLdG @ dG/dA
    dLdb = dLdG @ dG/db
    dLdc = dLdG @ dG/dc

    Args:
        G (np.ndarray): Fock representation of the Gaussian tensor
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the Fock representation G

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: derivatives of the loss with respect to A, b, c
    """
    shape = G.shape
    G = G.flatten()
    dLdG = dLdG.flatten()
    D = int(np.sqrt(G[1].size))
    N = G.shape[0]

    # calculate strides
    shape_arr = np.array(shape)
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]

    # init Jacobians
    dLdA = np.zeros((D, D), dtype=np.complex128)
    dLdb = np.zeros(D, dtype=np.complex128)
    dLdc = 0j

    # initialize path iterator
    path = np.ndindex(shape)

    # write vacuum amplitude derivatives
    dLdc += dLdG[next(path)]

    # initialize the flat grad array
    dG = np.zeros_like(dLdG)
    dG[0] = dLdG[0]

    idx = 0
    for nd_idx in path:
        idx += 1
        num_pivots = 0
        for i in range(D):
            if nd_idx[i] == 0:
                continue  # pivot would be out of bounds
            num_pivots += 1

        # from the formula G_k = sum_i G_k^i / N where G_k^i is the contribution from the i-th pivot
        # dL/dG_k^i = dL/dG_k * dG_k/dG_k^i = dL/dG_k * 1/N
        dLdG_k_i = dLdG[idx] / num_pivots
        for i in range(D):
            if nd_idx[i] == 0:
                continue  # pivot would be out of bounds

            pivot = idx - strides[i]

            # dL/dG_{k-1_i} = dL/dG_k^i * dG_k^i/dG_{k-1_i}
            # = dL/dG_k^i * b_i / sqrt(k_i)
            dG[pivot] += dLdG_k_i * b[i] / SQRT[nd_idx[i]]

            # dL/db_i = dL/dG_k^i * dG_k^i/db_i
            # = dL/dG_k^i * G_{k-1_i} / sqrt(k_i)
            dLdb[i] += dLdG_k_i * G[pivot] / SQRT[nd_idx[i]]

            # For A, we split j<i, j=i, j>i
            for j in range(i):
                neighbour = pivot - strides[j]
                # dL/dG_{k-1_i-1_j} = dL/dG_k^i * dG_k^i/dG_{k-1_i-1_j}
                # = dL/dG_k^i * A_{ij} * sqrt(k_j) / sqrt(k_i)
                dG[neighbour] += dLdG_k_i * A[i, j] * SQRT[nd_idx[j]] / SQRT[nd_idx[i]]

                # dL/dA_{ij} = dL/dG_k^i * dG_k^i/dA_{ij}
                # = dL/dG_k^i * G_{k-1_i-1_j} * sqrt(k_j) / sqrt(k_i)
                dLdA[i, j] += dLdG_k_i * G[neighbour] * SQRT[nd_idx[j]] / SQRT[nd_idx[i]]

            # j=i
            neighbour = pivot - strides[i]
            # dL/dG_{k-1_i-1_i} = dL/dG_k^i * dG_k^i/dG_{k-1_i-1_i}
            # = dL/dG_k^i * A_{ii} * sqrt(k_i-1) / sqrt(k_i)
            dG[neighbour] += dLdG_k_i * A[i, i] * SQRT[nd_idx[i] - 1] / SQRT[nd_idx[i]]

            # dL/dA_{ii} = dL/dG_k^i * dG_k^i/dA_{ii}
            # = dL/dG_k^i * G_{k-1_i-1_i} * sqrt(k_i-1) / sqrt(k_i)
            dLdA[i, i] += dLdG_k_i * G[neighbour] * SQRT[nd_idx[i] - 1] / SQRT[nd_idx[i]]

            for j in range(i + 1, D):
                neighbour = pivot - strides[j]
                # dL/dG_{k-1_i-1_j} = dL/dG_k^i * dG_k^i/dG_{k-1_i-1_j}
                # = dL/dG_k^i * A_{ij} * sqrt(k_j) / sqrt(k_i)
                dG[neighbour] += dLdG_k_i * A[i, j] * SQRT[nd_idx[j]] / SQRT[nd_idx[i]]

                # dL/dA_{ij} = dL/dG_k^i * dG_k^i/dA_{ij}
                # = dL/dG_k^i * G_{k-1_i-1_j} * sqrt(k_j) / sqrt(k_i)
                dLdA[i, j] += dLdG_k_i * G[neighbour] * SQRT[nd_idx[j]] / SQRT[nd_idx[i]]

    dLdc += dG[0] * c  # final gradient update for c
    return dLdA, dLdb, dLdc


@njit(parallel=True)
def vanilla_full_batch_vjp(
    G: ComplexTensor, c: ComplexVector, dLdG: ComplexTensor
) -> tuple[ComplexTensor, ComplexMatrix, ComplexVector]:  # pragma: no cover
    r"""Vector-Jacobian product (VJP) for the ``vanilla_full_batch`` function.
    Returns dL/dA, dL/db, dL/dc by parallelizing the single-instance ``vanilla_vjp`` over the batch dimension.

    Args:
        G (np.ndarray): Tensor result of the forward pass with shape `(batch_size,) + shape`.
        c (np.ndarray): Batched vacuum amplitudes with shape `(batch_size,)`.
        dLdG (np.ndarray): Gradient of the loss with respect to the output tensor `G`, with the same shape as `G`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: dL/dA, dL/db, dL/dc.
            dL/dA has shape `(batch_size, D, D)`, dL/db has shape `(batch_size, D)`, dL/dc has shape `(batch_size,)`.
            Where D is the number of modes (last dimension of G).
    """
    batch_size = G.shape[0]
    D = G.ndim - 1
    dLdA = np.zeros((batch_size, D, D), dtype=np.complex128)
    dLdb = np.zeros((batch_size, D), dtype=np.complex128)
    dLdc = np.zeros(batch_size, dtype=np.complex128)

    for k in prange(batch_size):
        dLdA_k, dLdb_k, dLdc_k = vanilla_vjp(G[k], c[k], dLdG[k])
        dLdA[k] = dLdA_k
        dLdb[k] = dLdb_k
        dLdc[k] = dLdc_k

    return dLdA, dLdb, dLdc
