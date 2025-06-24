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

"""neighbours functions"""

from collections.abc import Iterator

from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem

#################################################################################
## All neighbours means all the indices that differ from the given pivot by ±1 ##
#################################################################################


@njit
def all_neighbors(pivot: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
    r"""yields the indices of all the neighbours of the given index."""
    for j in range(len(pivot)):
        yield j, tuple_setitem(pivot, j, pivot[j] - 1)
        yield j, tuple_setitem(pivot, j, pivot[j] + 1)


####################################################################################
## Lower neighbours means all the indices that differ from the given index by -1  ##
####################################################################################


@njit
def lower_neighbors(pivot: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
    r"""yields the indices of the lower neighbours of the given index."""
    for j in range(len(pivot)):
        yield j, tuple_setitem(pivot, j, pivot[j] - 1)


####################################################################################
## Upper neighbours means all the indices that differ from the given index by +1  ##
####################################################################################


@njit
def upper_neighbors(pivot: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
    r"""yields the indices of the lower neighbours of the given index."""
    for j in range(len(pivot)):
        yield j, tuple_setitem(pivot, j, pivot[j] + 1)


####################################################################################################
## bitstring neighbours are indices that differ from the given index by ±1 according to a bitstring
####################################################################################################


@njit
def bitstring_neighbors(
    pivot: tuple[int, ...],
    bitstring: tuple[int, ...],
) -> Iterator[tuple[int, tuple[int, ...]]]:
    r"yields the indices of the bitstring neighbours of the given index"
    for i, b in enumerate(bitstring):
        if b:  # b == 1 -> lower
            yield i, tuple_setitem(pivot, i, pivot[i] - 1)
        else:  # b == 0 -> upper
            yield i, tuple_setitem(pivot, i, pivot[i] + 1)
