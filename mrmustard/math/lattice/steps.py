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

import numpy as np
from numba import njit, types

from mrmustard.math.lattice.neighbours import lower_neighbors_tuple
from mrmustard.math.lattice.pivots import first_pivot_tuple
from mrmustard.typing import ComplexMatrix, ComplexTensor, ComplexVector, Tensor

# TODO: gradients

SQRT = np.sqrt(np.arange(10000))


@njit
def vanilla_step(data: Tensor, A, b, index: tuple[int, ...]) -> complex:
    r"""Fock-Bargmann recurrence relation step. Vanilla version.
    This function calculates the index `index` of `tensor`.
    The appropriate pivot and neighbours must exist.

    Args:
        data (array or dict): fock amplitudes data store that supports getitem[tuple[int, ...]]
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (Sequence): index of the amplitude to calculate
    Returns:
        complex: the value of the amplitude at the given index
    """
    # index -> pivot
    i, pivot = first_pivot_tuple(index)

    # calculate value at index: pivot contribution
    denom = SQRT[pivot[i] + 1]
    value_at_index = b[i] / denom * data[pivot]

    # neighbors contribution
    for j, neighbor in lower_neighbors_tuple(pivot):
        value_at_index += A[i, j] / denom * SQRT[pivot[j]] * data[neighbor]

    return value_at_index


@njit
def vanilla_step_dict(data: types.DictType, A, b, index: tuple[int, ...]) -> complex:
    r"""Fock-Bargmann recurrence relation step. Vanilla version.
    This function calculates the index `index` of `tensor`.
    The appropriate pivot and neighbours must exist.

    Args:
        data: dict(tuple[int,...],complex): fock amplitudes numba dict
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (Sequence): index of the amplitude to calculate
    Returns:
        complex: the value of the amplitude at the given index
    """
    # index -> pivot
    i, pivot = first_pivot_tuple(index)

    # calculate value at index: pivot contribution
    denom = SQRT[pivot[i] + 1]
    value_at_index = b[i] / denom * data[pivot]

    # neighbors contribution
    for j, neighbor in lower_neighbors_tuple(pivot):
        value_at_index += A[i, j] / denom * SQRT[pivot[j]] * data.get(neighbor, 0.0 + 0.0j)

    return value_at_index


@njit
def binomial_step(
    Z: ComplexTensor, A: ComplexMatrix, b: ComplexVector, subspace_indices: list[tuple[int, ...]]
) -> tuple[ComplexTensor, float]:
    r"""Binomial step (whole subspace), array version.
    Iterates over the indices in ``subspace_indices`` and updates the tensor ``Z``.
    Returns the updated tensor and the probability of the subspace.

    Args:
        Z (np.ndarray): Tensor to be filled
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        subspace_indices (list[tuple[int, ...]]): list of indices to be updated

    Returns:
        tuple[np.ndarray, float]: updated tensor and probability of the subspace
    """
    prob = 0.0

    for i in range(len(subspace_indices)):
        value = vanilla_step(Z, A, b, subspace_indices[i])
        Z[subspace_indices[i]] = value
        prob = prob + np.abs(value) ** 2

    return Z, prob


@njit
def binomial_step_dict(
    Z: types.DictType, A: ComplexMatrix, b: ComplexVector, subspace_indices: list[tuple[int, ...]]
) -> tuple[types.DictType, float]:
    r"""Binomial step (whole subspace), dictionary version.
    Iterates over the indices in ``subspace_indices`` and updates the dictionary ``Z``.
    Returns the updated dictionary and the probability of the subspace.

    Args:
        Z (types.DictType): Dictionary to be filled
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        subspace_indices (list[tuple[int, ...]]): list of indices to be updated

    Returns:
        tuple[types.DictType, float]: updated dictionary and probability of the subspace
    """
    prob = 0.0

    for i in range(len(subspace_indices)):
        value = vanilla_step_dict(Z, A, b, subspace_indices[i])
        Z[subspace_indices[i]] = value
        prob = prob + np.abs(value) ** 2

    return Z, prob
