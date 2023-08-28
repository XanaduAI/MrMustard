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

from __future__ import annotations

import numpy as np

from mrmustard.lab.representations.data.data import Data
from mrmustard.math import Math
from mrmustard.typing import Scalar, Vector

math = Math()


class ArrayData(Data):
    """Contains array-like data for certain Representation objects.

    Args:
        array (Vector) : data to be contained in the class
    """

    def __init__(self, array: Vector) -> None:
        self.array = array

    @property
    def cutoffs(self):
        return self.array.shape

    def __neg__(self) -> Data:
        return self.__class__(array=-self.array)

    def __eq__(self, other: ArrayData) -> bool:
        try:
            return np.allclose(self.array, other.array)
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e

    def __add__(self, other: ArrayData) -> ArrayData:
        try:
            return self.__class__(array=self.array + other.array)
        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, other: Scalar) -> ArrayData:
        try:
            return self.__class__(array=self.array / other)
        except TypeError as e:
            raise TypeError("Can only divide by a scalar.") from e

    def __mul__(self, other: Scalar) -> ArrayData:
        try:
            return self.__class__(array=self.array * other)
        except TypeError as e:
            raise TypeError("Can only multiply by a scalar.") from e

    def __and__(self, other: ArrayData) -> ArrayData:
        try:
            return self.__class__(array=np.outer(self.array, other.array))
        except AttributeError as e:
            raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e
