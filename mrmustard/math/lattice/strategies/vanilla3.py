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

import numpy as np
from numba import njit

from mrmustard.math.lattice import paths, steps
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector
from .indices2 import first_available_pivot, lower_neighbours, project, shape_to_strides, shape_to_range

__all__ = ["vanilla", "vanilla_jacobian", "vanilla_vjp"]


@njit
def vanilla(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy. Fills the tensor by iterating over all indices
    in ndindex order.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    np_shape = np.array(shape)
    rng = shape_to_range(np_shape)
    strides = shape_to_strides(np_shape)

    # init output tensor
    ret = np.array([0 + 0j] * np.prod(np_shape), dtype=np.complex128)

    # write vacuum amplitude
    ret[0] = c

    # iterate over the rest of the indices
    for index in range(1, rng):
        i, pivot = first_available_pivot(index, strides)
        value_at_index = b[i] * ret[pivot]

        for (j, n) in lower_neighbours(pivot, strides):
            value_at_index += A[i, j] * np.sqrt(project(pivot, j, strides)) * ret[n]
        ret[index] = value_at_index/np.sqrt(project(index, i, strides))

    return ret.reshape(shape)
    # return np.array(ret).reshape(shape)



@njit
def vanilla_jacobian(
    G, A, b, c
) -> tuple[ComplexTensor, ComplexTensor, ComplexTensor]:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dG/dA, dG/db, dG/dc.
    Notice that G is a holomorphic function of A, b, c. This means that there is only
    one gradient to care about for each parameter (i.e. not dG/dA.conj() etc).
    """

    # init output tensors
    dGdA = np.zeros(G.shape + A.shape, dtype=np.complex128)
    dGdb = np.zeros(G.shape + b.shape, dtype=np.complex128)
    dGdc = G / c

    # initialize path iterator
    path = paths.ndindex_path(G.shape)

    # skip first index
    next(path)

    # iterate over the rest of the indices
    for index in path:
        dGdA, dGdb = steps.vanilla_step_jacobian(G, A, b, index, dGdA, dGdb)

    return dGdA, dGdb, dGdc


@njit
def vanilla_vjp(G, c, dLdG) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    D = G.ndim

    # init gradients
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize path iterator
    path = np.ndindex(G.shape)

    # skip first index
    next(path)

    # iterate over the rest of the indices
    for index in path:
        dA, db = steps.vanilla_step_grad(G, index, dA, db)
        dLdA += dA * dLdG[index]
        dLdb += db * dLdG[index]

    dLdc = np.sum(G * dLdG) / c

    return dLdA, dLdb, dLdc
