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

from typing import Iterator, Optional

import numpy as np
from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.typing import IntVector

# Strategies are generators of indices that follow paths in an N-dim positive integer lattice.
# The paths can cover the entire lattice, or just a subset of it.
# Strategies have to be generators because they enumerate lots of indices
# and we don't want to allocate memory for all of them at once.
# These strategies don't even reallocate memory for each index: they just
# yield the same array over and over again, modified each time, beware!


@njit
def zero_tuple(model: tuple[int, ...]) -> tuple[int, ...]:
    if len(model) == 0:
        return ()
    elif len(model) == 1:
        return (0,)
    elif len(model) == 2:
        return (0, 0)
    elif len(model) == 3:
        return (0, 0, 0)
    elif len(model) == 4:
        return (0, 0, 0, 0)
    elif len(model) == 5:
        return (0, 0, 0, 0, 0)
    elif len(model) == 6:
        return (0, 0, 0, 0, 0, 0)
    elif len(model) == 7:
        return (0, 0, 0, 0, 0, 0, 0)
    elif len(model) == 8:
        return (0, 0, 0, 0, 0, 0, 0, 0)
    elif len(model) == 9:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    elif len(model) == 10:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    elif len(model) == 11:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    else:
        raise ValueError("model too long")


@njit
def ndindex_path(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    r"yields the indices of a tensor in row-major order"
    index = tuple_setitem(shape, 0, 0)
    for i in range(1, len(shape)):
        index = tuple_setitem(index, i, 0)
    while True:
        yield index
        for i in range(len(shape) - 1, -1, -1):
            if index[i] < shape[i] - 1:
                index = tuple_setitem(index, i, index[i] + 1)
                break
            index = tuple_setitem(index, i, 0)
            if i == 0:
                return


@njit
def _find_combinations(cutoffs, total, index, current, result):
    if index == len(cutoffs):
        if total == 0:
            result.append(current)
        return

    for value in range(cutoffs[index]):
        if total - value >= 0:
            new_current = tuple_setitem(current, index, value)
            _find_combinations(cutoffs, total - value, index + 1, new_current, result)


@njit
def find_combinations(cutoffs, total):
    result = [cutoffs]
    _find_combinations(cutoffs, total, 0, cutoffs, result)
    return result[1:]


@njit
def equal_weight_path(
    shape: tuple[int, ...], max_sum: Optional[int] = None
) -> Iterator[list[tuple[int, ...]]]:
    r"""yields the indices of a tensor with equal weight.
    Effectively, `shape` contains local cutoffs (the maximum value of each index)
    and `max_sum` is the global cutoff (the maximum sum of all indices).
    If `max_sum` is not given, only the local cutoffs are used and the iterator
    yields  all possible indices within the tensor shape. In this case it becomes
    like `ndindex_iter` just in a different order.
    """
    if max_sum is None:
        max_sum = sum(shape) - len(shape)
    for s in range(max_sum + 1):
        yield find_combinations(shape, s)


@njit
def grey_code_iter(shape: IntVector) -> Iterator[IntVector]:
    raise NotImplementedError("Grey code order strategy not implemented yet")


@njit
def wormhole(shape: IntVector) -> IntVector:
    raise NotImplementedError("Wormhole strategy not implemented yet")


@njit
def diagonal(shape: IntVector) -> IntVector:
    raise NotImplementedError("Diagonal strategy not implemented yet")


@njit
def dynamic_U(shape: IntVector) -> IntVector:
    raise NotImplementedError("Diagonal strategy not implemented yet")
