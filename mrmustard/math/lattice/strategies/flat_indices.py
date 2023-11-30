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

from typing import Iterator, Sequence
from numba import njit

import numpy as np

@njit
def first_available_pivot(index: int, strides: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    r"""
    """
    for i in range(len(strides)):
        y = index - strides[i]
        if y >= 0:
            return (i, y)
    raise ValueError("Index is zero.")
    
@njit
def lower_neighbours(index: int, strides: Sequence[int], start: int) -> Iterator[tuple[int, tuple[int, ...]]]:
    for i in range(start, len(strides)):
        yield i, index - strides[i]

@njit 
def shape_to_strides(shape: Sequence[int]) -> Sequence[int]:
    r"""
    Calculates strides from shape.
    """
    strides = np.ones_like(shape)
    for i in range(1, len(shape)):
        strides[i-1] = np.prod(shape[i:])
    return strides
    