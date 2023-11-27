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
from numba import int64
from numba.experimental import jitclass

import numpy as np

__all__ = ["FlatIndex",]

spec = [
    ("_base", int64[:]),
    ("_range", int64),
    ("_shape", int64[:]),
    ("_value", int64),
]

@jitclass(spec)
class FlatIndex:
    r"""
    A class representing the index of a flattened tensor.
    """
    def __init__(self, shape: Sequence[int], value: int = 0) -> None:
        self._value = value
        self._shape = shape
        self._range = np.prod(shape)
        
        self._base = np.zeros_like(shape)
        for i in range(1, len(shape)):
            self._base[i-1] = np.prod(shape[i:])
        self._base[-1] = 1

    @property
    def base(self) -> Sequence[int]:
        r"""
        The base of this index.
        """
        return self._base
    
    @property
    def range(self) -> int:
        r"""
        The range of this index.
        """
        return self._range
    
    @property
    def shape(self) -> int:
        r"""
        The shape of this index.
        """
        return self._shape
        
    @property
    def value(self) -> int:
        r"""
        The value of this index.
        """
        return self._value
    
    def first_available_pivot(self):
        r"""
        """
        for (i, b) in enumerate(self.base):
            if self.value >= b:
                ret = FlatIndex(self.shape, self.value -b)
                return (i, ret)
        msg = "Index is zero."
        raise ValueError(msg)
    
    def increment(self) -> None:
        r"""
        Increments this index by ``1``.
        """
        self._value += 1
        if self.value >= self.range:
            msg = "FlatIndex cannot be incremented."
            raise ValueError(msg)
        
    def lower_neighbours(self) -> Sequence[int64]:
        for b in self.base:
            if self.value >= b:
                yield self.value - b
            else:
                yield self.value + self.range - b

    def __getitem__(self, idx: int64) -> int64:
        val = self.value
        for j in range(len(self.base)):
            bj = self.base[j]
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
    