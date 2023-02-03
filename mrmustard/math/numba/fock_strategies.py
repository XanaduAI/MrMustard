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

from typing import Generator, Tuple

import numpy as np
from numba import njit


@njit
def nearest_neighbours(index, shape):
    r"yields the indices of the nearest neighbours of the given index"
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
def lower_neighbours(index):
    r"yields the indices of the lower nearest neighbours of the given index"
    for i in range(len(index)):
        if index[i] > 0:
            index[i] -= 1
            yield index
            index[i] += 1


@njit
def upper_neighbours(index, shape):
    r"yields the indices of the upper nearest neighbours of the given index"
    for i in range(len(index)):
        if index[i] < shape[i]:
            index[i] += 1
            yield index
            index[i] -= 1


def vanilla_pivot(index):
    for i, v in enumerate(index):
        if v > 0:
            index[i] -= 1
            return index
    raise ValueError("Index is zero")


@njit
def strategy_ndindex(shape) -> Tuple[np.array, Generator]:
    index = np.zeros_like(shape)
    while True:
        yield index
        for i in range(len(shape) - 1, -1, -1):
            if index[i] < shape[i] - 1:
                index[i] += 1
                break
            else:
                index[i] = 0
                if i == 0:
                    return


@njit
def strategy_equal_weight(shape, max_sum=None) -> Tuple[np.array, Generator]:
    max_ = sum(shape) - len(shape) - 1
    max_sum = max_ if max_sum is None else min(max_sum, max_)
    for weight in range(max_sum + 1):
        index = np.zeros_like(shape)
        k = 0
        # first we distribute the weight over the indices
        while weight > 0:
            index[k] = weight if weight < shape[k] else shape[k] - 1
            weight -= index[k]
            k += 1
        # now we move units from the first index to the next index until we run out of units
        while True:
            yield index
            for i in range(len(index) - 1):
                if index[i] > 0 and index[i + 1] < shape[i + 1] - 1:
                    index[i] -= 1
                    index[i + 1] += 1
                    break
            else:
                break


@njit
def strategy_grey_code_order(shape) -> Generator:
    raise NotImplementedError("Grey code order strategy not implemented yet")


@njit
def wormhole(shape) -> Generator:
    raise NotImplementedError("Wormhole strategy not implemented yet")
