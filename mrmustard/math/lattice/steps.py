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

from mrmustard.math.lattice.neighbours import lower_neighbors_tuple
from mrmustard.math.lattice.pivots import first_pivot_tuple


# TODO: gradients

SQRT = np.sqrt(np.arange(10000))


@njit
def vanilla_step(tensor, A, b, index: tuple[int, ...]) -> complex:
    r"""Fock-Bargmann recurrence relation step. Vanilla version.
    This function calculates the index `index` of `tensor`.
    The appropriate pivot and neighbours must exist.

    Args:
        tensor (array): tensor to calculate the amplitudes of
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
    value_at_index = b[i] / denom * tensor[pivot]

    # neighbors contribution
    for j, neighbor in lower_neighbors_tuple(pivot):
        value_at_index += A[i, j] / denom * SQRT[pivot[j]] * tensor[neighbor]

    return value_at_index
