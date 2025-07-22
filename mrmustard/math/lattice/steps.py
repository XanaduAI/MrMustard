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

"""Fock-Bargmann recurrence relation steps for Gaussian states."""

import numpy as np
from numba import njit, prange, types
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.math.lattice.neighbors import lower_neighbors
from mrmustard.math.lattice.pivots import first_available_pivot
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

SQRT = np.sqrt(np.arange(100000))  # precompute sqrt of the first 100k integers


@njit(cache=True)
def vanilla_step(
    G: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexVector,
    index: tuple[int, ...],
) -> complex:  # pragma: no cover
    r"""Fock-Bargmann recurrence relation step, vanilla version.
    This function returns the amplitude of the Gaussian tensor G
    at G[index]. It does not modify G.
    The necessary pivot and neighbours must have already been computed,
    as this step will read those values from G.

    Args:
        G (array or dict): fock amplitudes data store that supports getitem[tuple[int, ...]]
        A (array): A matrix of the Fock-Bargmann representation
        b (array): B vector of the Fock-Bargmann representation
        index (Sequence): index of the amplitude to calculate
    Returns:
        complex: the value of the amplitude at the given index
    """
    # get pivot
    i, pivot = first_available_pivot(index)

    # pivot contribution
    value_at_index = b[i] * G[pivot]

    # neighbors contribution
    for j, neighbor in lower_neighbors(pivot):
        value_at_index += A[i, j] * SQRT[pivot[j]] * G[neighbor]

    return value_at_index / SQRT[index[i]]


@njit(cache=True)
def vanilla_step_batch(
    G: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexTensor,
    index: tuple[int, ...],
) -> complex:  # pragma: no cover
    r"""Fock-Bargmann recurrence relation step, batched version.
    This function returns the amplitude of the Gaussian tensor G
    at G[index]. It does not modify G.
    The necessary pivot and neighbours must have already been computed,
    as this step will read those values from G.
    Note that this function is different from vanilla_step as b is no longer a vector,
    it becomes a batched vector with the batch dimension on the first index.

    Args:
        G: fock amplitudes data store that supports getitem[tuple[int, ...]]
        A: A matrix of the Fock-Bargmann representation
        b: batched B vector of the Fock-Bargmann representation, the batch dimension is on the first index
        index: index of the amplitude to calculate
    Returns:
        array: the value of the amplitude at the given index according to each batch on the first index
    """
    i, pivot = first_available_pivot(index)

    # contribution from G[pivot]
    value_at_index = b[..., i] * G[(slice(None), *pivot)]
    # contributions from G[neighbor]
    for j, neighbor in lower_neighbors(pivot):
        value_at_index += A[..., i, j] * SQRT[pivot[j]] * G[(slice(None), *neighbor)]

    return value_at_index / SQRT[index[i]]


@njit(cache=True)
def vanilla_step_jacobian(
    G: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexVector,
    index: tuple[int, ...],
    dGdA: ComplexTensor,
    dGdB: ComplexTensor,
) -> tuple[ComplexTensor, ComplexTensor]:
    r"""Jacobian contribution of a single Fock-Bargmann recurrence relation step, vanilla version.
    It updates the dGdB and dGdA tensors at the given index.

    Args:
        G (array or dict): fully computed store that supports __getitem__(index: tuple[int, ...])
        A (array): A matrix of the Fock-Bargmann representation
        b (array): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        index (Sequence): index at which to compute the jacobian
        dGdB (array): gradient of G with respect to b (partially computed)
        dGdA (array): gradient of G with respect to A (partially computed)
    Returns:
        tuple[array, array]: the dGdB and dGdA tensors updated at the given index
    """
    # index -> pivot
    i, pivot = first_available_pivot(index)

    # pivot contribution
    dGdB[index] += b[i] * dGdB[pivot] / SQRT[index[i]]
    dGdB[(*index, i)] += G[pivot] / SQRT[index[i]]
    dGdA[index] += b[i] * dGdA[pivot] / SQRT[index[i]]

    # neighbors contribution
    for j, neighbor in lower_neighbors(pivot):
        dGdB[index] += A[i, j] * dGdB[neighbor] * SQRT[pivot[j]] / SQRT[index[i]]
        dGdA[index] += A[i, j] * dGdA[neighbor] * SQRT[pivot[j]] / SQRT[index[i]]
        dGdA[(*index, i, j)] += G[neighbor] * SQRT[pivot[j]] / SQRT[index[i]]

    return dGdA, dGdB


@njit(cache=True)
def vanilla_step_grad(
    G: ComplexTensor,
    index: tuple[int, ...],
    dA: ComplexMatrix,
    db: ComplexVector,
) -> tuple[ComplexMatrix, ComplexVector]:
    r"""Gradient with respect to A and b of a single Fock-Bargmann recurrence relation step,
    vanilla version. dA and db can be used to update the dGdB and dGdA tensors at `index`,
    or as part of the contraction with the gradient of the Gaussian tensor G.
    Note that the gradient depends only on G and not on the values of A and b.

    Args:
        G (array or dict): fully computed store that supports __getitem__(index: tuple[int, ...])
        index (Sequence): index of the amplitude to calculate the gradient of
        dA (array): empty array to store the gradient of G[index] with respect to A
        db (array): empty array to store the gradient of G[index] with respect to B
    Returns:
        tuple[array, array]: the updated dGdB and dGdA tensors
    """
    for i in range(len(db)):
        pivot_i = tuple_setitem(index, i, index[i] - 1)
        db[i] = SQRT[index[i]] * G[pivot_i]
        dA[i, i] = 0.5 * SQRT[index[i] * pivot_i[i]] * G[tuple_setitem(pivot_i, i, pivot_i[i] - 1)]
        for j in range(i + 1, len(db)):
            dA[i, j] = SQRT[index[i] * pivot_i[j]] * G[tuple_setitem(pivot_i, j, pivot_i[j] - 1)]

    return dA, db


@njit(cache=True)
def vanilla_step_dict(
    data: types.DictType,
    A: ComplexMatrix,
    b: ComplexVector,
    index: tuple[int, ...],
) -> complex:
    r"""Fock-Bargmann recurrence relation step, vanilla version with numba dict.
    This function calculates the index `index` of the Gaussian tensor `G`.
    The appropriate pivot and neighbours must exist.

    Args:
        data: dict(tuple[int,...], complex): fock amplitudes numba dict
        A (array): A matrix of the Fock-Bargmann representation
        b (array): b vector of the Fock-Bargmann representation
        index (Sequence): index of the amplitude to calculate
    Returns:
        complex: the value of the amplitude at the given index
    """
    # index -> pivot
    i, pivot = first_available_pivot(index)

    # calculate value at index: pivot contribution
    denom = SQRT[pivot[i] + 1]
    value_at_index = b[i] / denom * data[pivot]

    # neighbors contribution
    for j, neighbor in lower_neighbors(pivot):
        value_at_index += A[i, j] / denom * SQRT[pivot[j]] * data.get(neighbor, 0.0 + 0.0j)

    return value_at_index


@njit(cache=True)
def binomial_step(
    G: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexVector,
    subspace_indices: list[tuple[int, ...]],
) -> tuple[ComplexTensor, float]:
    r"""Computes a whole subspace of the ``G`` tensor at the indices in
    ``subspace_indices`` (a subspace is such that `sum(index) = const`).
    It updates the tensor ``G``.
    It returns the updated tensor and the probability of the subspace.

    Args:
        G (np.ndarray): Tensor filled up to the previous subspace
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        subspace_indices (list[tuple[int, ...]]): list of indices to be updated

    Returns:
        tuple[np.ndarray, float]: updated tensor and probability of the subspace
    """
    norm = 0.0

    for i in prange(len(subspace_indices)):
        value = vanilla_step(G, A, b, subspace_indices[i])
        G[subspace_indices[i]] = value
        norm = norm + np.abs(value) ** 2

    return G, norm


@njit(cache=True)
def binomial_step_dict(
    G: types.DictType,
    A: ComplexMatrix,
    b: ComplexVector,
    subspace_indices: list[tuple[int, ...]],
) -> tuple[types.DictType, float]:
    r"""Computes a whole subspace of the ``G`` dict at the indices in
    ``subspace_indices`` (a subspace is such that `sum(index) = const`).
    It updates the dict ``G``. It returns the updated G dict
    and the total probability of the subspace.

    Args:
        G (types.DictType): Dictionary filled up to the previous subspace
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        subspace_indices (list[tuple[int, ...]]): list of indices to be updated

    Returns:
        tuple[types.DictType, float]: updated dictionary and probability of the subspace
    """
    prob = 0.0

    for i in prange(len(subspace_indices)):
        value = vanilla_step_dict(G, A, b, subspace_indices[i])
        G[subspace_indices[i]] = value
        prob = prob + np.abs(value) ** 2

    return G, prob
