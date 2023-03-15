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

from typing import Iterator

import numpy as np
from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.typing import Batch, IntMatrix, IntVector

#################################################################################
## All neighbours means all the indices that differ from the given pivot by ±1 ##
#################################################################################


@njit
def all_neighbours_iter(pivot: IntVector) -> Iterator[IntVector]:
    r"yields the indices of all the nearest neighbours of the given pivot"
    for i in range(len(pivot)):
        pivot[i] += 1
        yield pivot
        pivot[i] -= 2
        yield pivot
        pivot[i] += 1


@njit
def all_neighbours_fn(pivot: IntVector, Z: IntMatrix) -> Batch[IntVector]:
    r"returns the indices of the nearest neighbours of the given pivot as an array"
    for i, _ in enumerate(pivot):
        Z[2 * i] = pivot
        Z[2 * i + 1] = pivot
        Z[2 * i, i] += 1
        Z[2 * i + 1, i] -= 1
    return Z


####################################################################################
## Lower neighbours means all the indices that differ from the given index by -1  ##
####################################################################################


@njit
def lower_neighbors_fn(pivot: IntVector, Z: IntMatrix) -> Batch[IntVector]:
    r"returns the indices of the lower neighbours of the given index as an array"
    for j, _ in enumerate(pivot):
        Z[j] = pivot
        Z[j, j] -= 1
    return Z


@njit
def lower_neighbors(pivot: IntVector) -> Iterator[tuple[int, IntVector]]:
    r"""yields the indices of the lower neighbours of the given index.
    Modifies the index in place.
    """
    for j in range(len(pivot)):
        pivot[j] -= 1
        yield j, pivot
        pivot[j] += 1


@njit
def lower_neighbors_tuple(pivot: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
    r"""yields the indices of the lower neighbours of the given index."""
    for j in range(len(pivot)):
        yield j, tuple_setitem(pivot, j, pivot[j] - 1)


####################################################################################
## Upper neighbours means all the indices that differ from the given index by +1  ##
####################################################################################


@njit
def upper_neighbors(pivot: IntVector) -> Iterator[IntVector]:
    r"yields the indices of the upper neighbours of the given pivot"
    for i in range(len(pivot)):
        pivot[i] += 1
        yield pivot
        pivot[i] -= 1


@njit
def upper_neighbors_fn(pivot: IntVector, Z: IntMatrix) -> Batch[IntVector]:
    r"returns the indices of the upper neighbours of the given index as an array"
    for i, _ in enumerate(pivot):
        Z[i] = pivot
        Z[i, i] += 1
    return Z


####################################################################################################
## bitstring neighbours are indices that differ from the given index by ±1 according to a bitstring
####################################################################################################


@njit
def bitstring_neighbours_iter(pivot: IntVector, bitstring: IntVector) -> Iterator[IntVector]:
    r"yields the indices of the bitstring neighbours of the given index"
    for i, b in enumerate(bitstring):
        if b:  # b == 1 -> subtract 1
            pivot[i] -= 1
            yield pivot
            pivot[i] += 1
        else:  # b == 0 -> add 1
            pivot[i] += 1
            yield pivot
            pivot[i] -= 1
