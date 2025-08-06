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
#

from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem


@njit(cache=True)
def first_available_pivot(index: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    r"""returns the first available pivot for the given index. A pivot is a nearest neighbor
    of the index. Here we pick the first available pivot.

    Arguments:
        index: the index to get the first available pivot of.

    Returns:
        the index that was decremented and the pivot
    """
    for i, v in enumerate(index):
        if v > 0:
            return i, tuple_setitem(index, i, v - 1)
    raise ValueError("Index is zero")


@njit(cache=True)
def smallest_pivot(index: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    r"""returns the pivot closest to a zero index. A pivot is a nearest neighbor
    of the index. Here we pick the pivot with the smallest non-zero element.

    Arguments:
        index: the index to get the smallest pivot of.

    Returns:
        (int, tuple) the index of the element that was decremented and the pivot
    """
    min_ = 2**64 - 1
    for i, v in enumerate(index):
        if 0 < v < min_:
            min_ = v
            min_i = i
    if min_ == 2**64 - 1:
        raise ValueError("Index is zero")
    return min_i, tuple_setitem(index, min_i, min_ - 1)


@njit(cache=True)
def all_pivots(
    index: tuple[int, ...],
) -> list[tuple[int, tuple[int, ...]]]:
    r"""returns all the pivots for the given index. A pivot is a nearest neighbor
    of the index one index lowered.

    Arguments:
        index: the index to get the pivots of.

    Returns:
        a list of the indices that were decremented and the pivots
    """
    return [(i, tuple_setitem(index, i, v - 1)) for i, v in enumerate(index) if v > 0]
