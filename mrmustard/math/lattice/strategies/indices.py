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

from typing import Sequence
from numba import int64, njit
from numba.experimental import jitclass

import numpy as np

__all__ = ["FlatIndex", "shape_to_strides", "shape_to_range"]

spec = [
    ("_value", int64),
    ("_range", int64),
    ("_strides", int64[:]),
]

@jitclass(spec)
class FlatIndex:
    r"""
    A class representing the index of a flattened tensor.
    """
    def __init__(self, value: int, range: int, strides: Sequence[int]) -> None:
        self._value = value
        self._range = range
        self._strides = strides        

    @property
    def strides(self) -> Sequence[int]:
        r"""
        The strides of this index.
        """
        return self._strides
    
    @property
    def range(self) -> int:
        r"""
        The range of this index.
        """
        return self._range
        
    @property
    def value(self) -> int:
        r"""
        The value of this index.
        """
        return self._value
    
    def first_available_pivot(self):
        r"""
        """
        for (i, b) in enumerate(self.strides):
            if self.value >= b:
                ret = FlatIndex(self.value - b, self.range, self.strides)
                return (i, ret)
        msg = "Index is zero."
        raise ValueError(msg)
    
    def increment(self) -> None:
        r"""
        Increments this index by ``1``.
        """
        self._value += 1
        # if self.value >= self.range:
        #     msg = "``FlatIndex`` cannot be incremented."
        #     raise ValueError(msg)
        
    def lower_neighbours(self) -> tuple[int, Sequence[int]]:
        for (j, b) in enumerate(self.strides):
            y = self.value - b
            # yield j, y if y >= 0 else y + self.range
            if y >= 0:
                yield j, y

    def __getitem__(self, idx: int) -> int:
        val = self.value
        for j in range(len(self.strides)):
            bj = self.strides[j]
            if idx == j:
                ret = 0
                while val >= bj:
                    ret += 1
                    val -= bj
                return ret
            while val >= bj:
                val -= bj
        msg = "Cannot find element ``idx`` in FlatIndex."
        raise ValueError(msg)

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
    ret = np.prod(shape)
    return ret
    