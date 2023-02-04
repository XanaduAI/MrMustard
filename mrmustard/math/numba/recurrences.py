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

from mrmustard.types import Int1D, Int2D, Matrix, Tensor, Vector

from .neighbours import lower_neighbors
from .pivots import vanilla_pivot


def general_step(
    tensor: Tensor,
    A: Matrix,
    b: Vector,
    index: Int1D,
    pivot_fn: Callable[[Int1D], Int1D],
    neighbors_fn: Callable[[Int1D], Int2D],
    Ab_fn: Callable[
        [Matrix, Vector, Int1D, Callable[[Int1D], Int2D]], [Matrix, Vector]
    ] = lambda A, b, pivot, neighbors_fn: (A, b),
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
    pivot = pivot_fn(index)
    A = A * np.sqrt(np.asarray(pivot))[None, :] / np.sqrt(np.asarray(pivot) + 1)[:, None]
    b = b / np.sqrt(np.asarray(pivot) + 1)
    A, b = Ab_fn(A, b, pivot)
    return b * tensor[pivot] + A @ tensor[neighbors_fn(pivot)]


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
    return general_step(tensor, A, b, index, vanilla_pivot, lower_neighbors)


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
    return general_step(tensor, A, b, index, vanilla_pivot, lower_neighbors)


#%%
import mpmath
import numpy as np

a = np.array(
    [
        [mpmath.mpf(0.2234, prec=100), mpmath.mpf(0.2345, prec=100)],
        [mpmath.mpf(0.567, prec=100), mpmath.mpf(0.5678, prec=100)],
    ]
)

from time import time

t0 = time()
for i in range(10000):
    a ** (-1)
print((time() - t0) / 10000)
# %%
