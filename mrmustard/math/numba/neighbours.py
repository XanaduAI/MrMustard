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

import numpy as np
from numba import njit

from mrmustard.types import Generator, Int1D, Int2D

#################################################################################
## All neighbours means all the indices that differ from the given index by ±1 ##
#################################################################################


@njit
def all_neighbours_gen(index: Int1D, shape: Int1D) -> Generator[Int1D, None, None]:
    r"yields the indices of all the nearest neighbours of the given index"
    for i in range(len(index)):
        if index[i] < shape[i] - 1:
            index[i] += 1
            yield index
            index[i] -= 1
        if index[i] > 0:
            index[i] -= 1
            yield index
            index[i] += 1


@njit
def all_neighbours_fn(pivot: Int1D, shape: Int1D) -> Int2D:
    r"returns the indices of the nearest neighbours of the given index"
    Z = np.zeros((2 * len(pivot), len(pivot)), dtype=int)
    for i, p in enumerate(pivot):
        if p > 0:  # can get lower neighbour
            pivot[i] -= 1
            Z[2 * i] = pivot
            pivot[i] += 1
        if p < shape[i] - 1:  # can get upper neighbour
            pivot[i] += 1
            Z[2 * i + 1] = pivot
            pivot[i] -= 1
    return Z


####################################################################################
## Lower neighbours means all the indices that differ from the given index by -1  ##
####################################################################################


@njit
def lower_neighbors_gen(index: Int1D) -> Generator[Int1D, None, None]:
    r"yields the indices of the lower neighbours of the given index"
    for i in range(len(index)):
        if index[i] > 0:
            index[i] -= 1
            yield index
            index[i] += 1


@njit
def lower_neighbors_fn(pivot: Int1D) -> Int2D:
    r"returns the indices of the lower neighbours of the given index"
    Z = np.zeros((len(pivot), len(pivot)), dtype=int)
    for i, p in enumerate(pivot):
        pivot[i] += 1
        Z[i] = pivot
        pivot[i] -= 1
    return Z


####################################################################################
## Upper neighbours means all the indices that differ from the given index by +1  ##
####################################################################################


@njit
def upper_neighbors_gen(index: Int1D, shape: Int1D) -> Generator[Int1D, None, None]:
    r"yields the indices of the upper neighbours of the given index"
    for i in range(len(index)):
        if index[i] < shape[i]:
            index[i] += 1
            yield index
            index[i] -= 1


@njit
def upper_neighbors_fn(pivot: Int1D, shape: Int1D) -> Int2D:
    r"returns the indices of the upper neighbours of the given index"
    Z = np.zeros((len(pivot), len(pivot)), dtype=int)
    for i, p in enumerate(pivot):
        pivot[i] -= 1
        Z[i] = pivot
        pivot[i] += 1
    return Z
