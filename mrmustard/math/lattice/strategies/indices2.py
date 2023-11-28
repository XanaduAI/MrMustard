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
def first_available_pivot(index: int, strides: Sequence[int]):
    r"""
    """
    for (i, b) in enumerate(strides):
        if index >= b:
            return (i, index - b)
    raise ValueError("Index is zero.")
    
@njit
def lower_neighbours(index: int, strides: Sequence[int]) -> Iterator[tuple[int, tuple[int, ...]]]:
    for (i, b) in enumerate(strides):
        y = index - b
        if y >= 0:
            yield i, y

@njit
def project(index: int, idx: int, strides: Sequence[int]) -> int:
    for j, bj in enumerate(strides):
        if idx == j:
            ret = 0
            while index >= bj:
                ret += 1
                index -= bj
            return ret
        while index >= bj:
            index -= bj
    raise ValueError("Cannot find element ``idx`` in FlatIndex.")

@njit
def project_on_dominant_stride(index: int, idx: int, strides: Sequence[int]) -> int:
    return 1
    bj = strides[idx]
    ret = 0
    while index >= bj:
        ret += 1
        index -= bj
    return ret

@njit 
def shape_to_strides(shape: Sequence[int]) -> Sequence[int]:
    r"""
    Calculates strides from shape.
    """
    strides = np.ones_like(shape)
    for i in range(1, len(shape)):
        strides[i-1] = np.prod(shape[i:])
    return strides

@njit 
def shape_to_range(shape: Sequence[int]) -> int:
    r"""
    Calculates range from shape.
    """
    return np.prod(shape)
    