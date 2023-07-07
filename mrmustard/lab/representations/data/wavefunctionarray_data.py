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

from typing import Union

from mrmustard.lab.representations.data.array_data import ArrayData
from mrmustard.lab.representations.data.data import Data
from mrmustard.math import Math
from mrmustard.typing import Scalar
from mrmustard.utils.misc_tools import duck_type_checker

math = Math()


class WavefunctionArrayData(ArrayData):
    r"""Data class for the Wavefunction, encapsulating q-variable points and corresponding values.

    Args:
        qs:     q-variable points
        array:  q-Wavefunction values corresponding to the qs
    """

    def __init__(self, qs: np.array, array: np.array) -> None:
        super().__init__(array=array)
        self.qs = qs

    def __neg__(self) -> Data:
        return self.__class__(array=-self.array, qs=self.qs)

    def __eq__(self, other: ArrayData) -> bool:
        try:
            return super().same(X=[self.array], Y=[other.array]) and super().same(
                X=[self.qs], Y=[other.qs]
            )

        except AttributeError as e:
            raise TypeError(
                f"Cannot compare {self.__class__} and {other.__class__}."
            ) from e

    def __truediv__(self, x: Scalar) -> ArrayData:
        try:
            return self.__class__(array=self.array / x, qs=self.qs)
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Cannot divide {self.__class__} by {x}.") from e

    def __add__(self, other: ArrayData) -> WavefunctionArrayData:
        if self._qs_is_same(other):
            try:
                return self.__class__(
                    array=self.array + other.array, qs=self.qs
                )  # TODO

            except AttributeError as e:
                raise TypeError(
                    f"Cannot add/subtract {self.__class__} and {other.__class__}."
                ) from e
        else:
            raise ValueError("The two wave functions must have the same qs. ")

    def __sub__(self, other: WavefunctionArrayData) -> WavefunctionArrayData:
        try:
            if self._qs_is_same(other):
                return self.__add__(-other)
            else:
                raise ValueError("The two wave functions must have the same qs. ")
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Cannot subtract {self.__class__} and {other.__class__}."
            ) from e

    def __mul__(
        self, other: Union[Scalar, WavefunctionArrayData]
    ) -> WavefunctionArrayData:
        if duck_type_checker(self, other):  # same type
            if self._qs_is_same(other):
                new_array = self.array * other.array
                return self.__class__(array=new_array, qs=self.qs)
            else:
                raise ValueError("The two wave functions must have the same qs. ")
        else:  # assuming it's a scalar...
            try:
                new_array = self.array * other
                return self.__class__(array=new_array, qs=self.qs)
            except TypeError as e:  # ... and it wasn't a scalar!
                raise TypeError(
                    f"Cannot add/subtract {self.__class__} and {other.__class__}."
                ) from e

    def __and__(self, other: WavefunctionArrayData) -> WavefunctionArrayData:
        try:
            return self.__class__(
                array=np.outer(self.array, other.array), qs=np.outer(self.qs, other.qs)
            )
        except AttributeError as e:
            raise TypeError(
                f"Cannot tensor {self.__class__} and {other.__class__}."
            ) from e

    def _qs_is_same(self, other: WavefunctionArrayData) -> bool:
        r"""Compares the qs of two WavefunctionArrayData objects."""
        try:
            return True if np.allclose(self.qs, other.qs) else False
        except AttributeError as e:
            raise TypeError(
                f"Cannot compare {self.__class__} and {other.__class__}."
            ) from e
