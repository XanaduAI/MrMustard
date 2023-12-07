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

r"""
Contains the functions to operate with flattened indices.

Given a multi-dimensional ``np.ndarray``, we can index its elements using ``np.ndindex``.
Alternatevely, we can flatten the multi-dimensional array and index its elements with
``int``s (hereby referred to as ''flat indices''). 
"""

from typing import Iterator, Sequence
from numba import njit

import numpy as np


@njit
def first_available_pivot(
    index: int, strides: Sequence[int]
) -> tuple[int, tuple[int, ...]]:  # pragma: no cover
    r"""
    Returns the first available pivot for the given flat index.
    A pivot is a nearest neighbor of the index. Here we pick the first available pivot.

    Arguments:
        index: the flat index to get the first available pivot of.
        strides: the strides that allow mapping the flat index to a tuple index.

    Returns:
        the flat index that was decremented and the pivot.
    """
    for i, s in enumerate(strides):
        y = index - s
        if y >= 0:
            return (i, y)
    raise ValueError("Index is zero.")


@njit
def lower_neighbors(
    index: int, strides: Sequence[int], start: int
) -> Iterator[tuple[int, tuple[int, ...]]]:  # pragma: no cover
    r"""
    Yields the flat indices of the lower neighbours of the given flat index.
    """
    for i in range(start, len(strides)):
        yield i, index - strides[i]


@njit
def shape_to_strides(shape: Sequence[int]) -> Sequence[int]:  # pragma: no cover
    r"""
    Calculates strides from shape.

    Arguments:
        shape: the shape of the ``np.ndindex``.

    Returns:
        the strides that allow mapping a flat index to the corresponding ``np.ndindex``.
    """
    strides = np.ones_like(shape)
    for i in range(1, len(shape)):
        strides[i - 1] = np.prod(shape[i:])
    return strides
