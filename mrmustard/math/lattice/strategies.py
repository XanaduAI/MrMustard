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

from typing import Any, Generator, Optional

import numpy as np
from numba import njit

from mrmustard.types import Vector

# Strategies are generators of indices that follow paths in an N-dim positive integer lattice.
# The paths can cover the entire lattice, or just a subset of it.
# Strategies have to be generators because they enumerate lots of indices
# and we don't want to allocate memory for all of them at once.
# These strategies don't even reallocate memory for each index, they just
# return the same array over and over again. So anything pointing to it, like
# the entries of list(strategy), will be equal to the last index.


# @njit
def ndindex_gen(shape: Vector) -> Generator[Vector, None, None]:
    r"yields the indices of a tensor in row-major order"
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
def equal_weight_gen(shape: Vector, max_sum: Optional[int] = None) -> Generator[Vector, None, None]:
    r"yields the indices of a tensor with equal weight"
    max_ = sum(shape) - len(shape) - 1  # allows to fill the entire tensor
    max_sum = max_ if max_sum is None else min(max_sum, max_)
    for weight in range(max_sum + 1):
        index = np.zeros_like(shape)
        k = 0
        # first we distribute the weight over the indices
        while weight > 0:
            index[k] = weight if weight < shape[k] else shape[k] - 1
            weight -= index[k]
            k += 1
        # now we move units from the first index to the next until we run out
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
def grey_code_gen(shape: Vector) -> Generator[Vector, None, None]:
    raise NotImplementedError("Grey code order strategy not implemented yet")


@njit
def wormhole(shape: Vector) -> Any:
    raise NotImplementedError("Wormhole strategy not implemented yet")
