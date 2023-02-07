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

from typing import Callable, Tuple

import numpy as np

from mrmustard.types import Matrix, Tensor, Vector

from .neighbours import lower_neighbors_fn
from .pivots import first_pivot_fn

# TODO: gradients


# @njit
def general_step(
    tensor: Tensor,
    A: Matrix,
    b: Vector,
    index: Vector,
    pivot_fn: Callable[[Vector], Vector],
    neighbors_fn: Callable[[Vector], Matrix],
    Ab_fn: Callable = lambda A, b, neighbors_fn, pivot: (A, b),
):
    r"""Fock-Bargmann recurrence relation step. General version.
    Args:
        tensor (array): tensor to calculate the amplitudes of
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (tuple): index of the amplitude to calculate
        pivot_fn (callable): function that returns the pivot corresponding to the index
        neighbors_fn (callable): function that returns the neighbors of the pivot
    Returns:
        complex: the value of the amplitude at the given index
    """
    i, pivot = pivot_fn(index)
    A = A * np.sqrt(pivot)[None, :] / np.sqrt(pivot + 1)[:, None]
    b = b / np.sqrt(pivot + 1)
    A, b = Ab_fn(A, b, neighbors_fn, pivot)
    neighbors = neighbors_fn(pivot)  # neighbors is an array of indices len(pivot) x len(pivot)
    value_at_index = b[i] * tensor_value(tensor, pivot)
    print("-" * 80)
    print(f"tensor_value(tensor, index={pivot})={tensor_value(tensor, pivot)}")
    for j, neighbor in enumerate(neighbors):
        val = tensor_value(tensor, neighbor)
        print(f"tensor_value(tensor={tensor}, index={neighbor})={val}")
        value_at_index += A[i, j] * val
    return value_at_index


# @njit
def vanilla_step(tensor, A, b, index: Tuple) -> complex:
    """Fock-Bargmann recurrence relation step. Vanilla version.
    Args:
        tensor (array): tensor to calculate the amplitudes of
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (tuple): index of the amplitude to calculate
        pivot_fn (callable): function that returns the pivot corresponding to the index
        neighbors_fn (callable): function that returns the neighbors of the pivot
    Returns:
        complex: the value of the amplitude at the given index
    """
    return general_step(tensor, A, b, index.copy(), first_pivot_fn, lower_neighbors_fn)


### array to tuple functions ###


# @njit
def ravel_multi_index(index, shape):
    res = 0
    for i in range(len(index)):
        res += index[i] * np.prod(np.asarray(shape)[i + 1 :])
    return res


# @njit
def tensor_value(tensor, index):
    return tensor.flat[ravel_multi_index(index, tensor.shape)]
