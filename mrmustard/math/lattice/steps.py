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

from typing import Callable, Optional

import numpy as np
from numba import njit

from mrmustard.math.lattice.neighbours import lower_neighbors_fn
from mrmustard.math.lattice.pivots import first_pivot_fn
from mrmustard.math.lattice.utils import tensor_value
from mrmustard.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    IntVector,
    Matrix,
    Vector,
)

# TODO: gradients


@njit
def general_step(
    tensor: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexVector,
    index: IntVector,
    pivot_idx: IntVector,
    neighbors_idx: Batch[IntVector],
    pivot_fn: Callable[[IntVector], IntVector],
    neighbors_fn: Callable[[IntVector], Batch[IntVector]],
    Ab_fn: Optional[Callable] = None,
):
    r"""Fock-Bargmann recurrence relation step. General version.
    requires selecting a pivot and a set of neighbors.
    Args:
        tensor (array): tensor to calculate the amplitudes of
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (tuple): index of the amplitude to calculate
        pivot_fn (callable): function that returns the pivot corresponding to the index
        neighbors_fn (callable): function that returns the neighbors of the pivot
        Ab_fn (callable): function that returns the new values of A and b
    Returns:
        complex: the value of the amplitude at the given index
    """
    print(" " * 8 + "[general_step] called with index:", index)
    i, pivot = pivot_fn(index, pivot_idx)
    print(" " * 8 + "[general_step] i,pivot returned from index", index, (i, pivot))
    A = A * np.expand_dims(np.sqrt(pivot), 0) / np.expand_dims(np.sqrt(pivot + 1), 1)
    b = b / np.sqrt(pivot + 1)
    if Ab_fn is not None:
        A, b = Ab_fn(A, b, pivot, neighbors_fn)
    print(" " * 8 + "[general_step] about to call neighbors_fn")
    neighbors = neighbors_fn(
        pivot, neighbors_idx
    )  # neighbors is an array of indices len(pivot) x len(pivot)
    print(" " * 8 + "[general_step] neighbors obtained:", neighbors)
    print(" " * 8 + "[general_step] computing tensor_value(tensor, pivot)")
    value_at_index = b[i] * tensor_value(tensor, pivot)
    print(" " * 8 + "[general_step] tensor value:", tensor_value(tensor, pivot))
    for j, neighbor in enumerate(neighbors):
        print(" " * 8 + "[general_step] neighbor:", neighbor, " <-- pivot:", pivot)
        # value_at_index += tensor_value(tensor, neighbor)
        value_at_index += A[i, j] * tensor_value(tensor, neighbor)
    # index[i] += 1  # restore the index
    return value_at_index


@njit
def vanilla_step(tensor, A, b, index: Vector, pivot_idx: Vector, neighbors_idx: Matrix) -> complex:
    r"""Fock-Bargmann recurrence relation step. Vanilla version.
    This function calculates the index `index` of `tensor`.
    The appropriate pivot and neighbours must exist.

    Args:
        tensor (array): tensor to calculate the amplitudes of
        A (array): matrix of coefficients
        b (array): vector of coefficients
        index (Sequence): index of the amplitude to calculate
        pivot_fn (callable): function that returns the pivot corresponding to the index
        neighbors_fn (callable): function that returns the neighbors of the pivot
    Returns:
        complex: the value of the amplitude at the given index
    """
    print(" " * 4, "[vanilla_step] called with index:", index)
    print(" " * 4, "[vanilla_step] calling general_step")
    return general_step(
        tensor, A, b, index, pivot_idx, neighbors_idx, first_pivot_fn, lower_neighbors_fn
    )


### array to tuple functions ###
