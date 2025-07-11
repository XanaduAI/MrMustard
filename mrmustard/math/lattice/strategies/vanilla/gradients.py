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


@njit(cache=True)
def vanilla_vjp_numba(
    G,
    c,
    dLdG,
) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
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
    D = len(shape_arr)

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # linearize G
    G_lin = G.ravel()

    # init gradients
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize the n-dim index
    nd_index = np.ndindex(G.shape)
    next(nd_index)

    # iterate over the indices (no need to split the loop in two parts)
    for flat_index, index_u in enumerate(nd_index):
        # contributions from pivot and lower neighbours
        for i in range(D):
            pivot = (flat_index + 1) - strides[i]
            db[i] = SQRT[index_u[i]] * G_lin[pivot]
            dA[i, i] = (
                0.5 * SQRT[index_u[i]] * SQRT[index_u[i] - 1] * G_lin[pivot - strides[i]]
                if index_u[i] > 1
                else 0.0
            )
            for j in range(i + 1, D):
                dA[i, j] = SQRT[index_u[i]] * SQRT[index_u[j]] * G_lin[pivot - strides[j]]

        dLdA += dA * dLdG[index_u]
        dLdb += db * dLdG[index_u]

    dLdc = np.sum(G * dLdG) / c

    return (dLdA + dLdA.T) / 2, dLdb, dLdc


@njit(cache=True, parallel=True)
def vanilla_batch_vjp_numba(
    G: ComplexTensor,
    c: ComplexVector,
    dLdG: ComplexTensor,
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
        dLdA_k, dLdb_k, dLdc_k = vanilla_vjp_numba(G[k], c[k], dLdG[k])
        dLdA[k] = dLdA_k
        dLdb[k] = dLdb_k
        dLdc[k] = dLdc_k

    return dLdA, dLdb, dLdc
