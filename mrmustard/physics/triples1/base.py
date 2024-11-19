# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from mrmustard import math
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
)

__all__ = ["Coefficient"]


class Coefficient(ABC):
    r"""
    A base class for the coefficients in the PolyExpAnsatz.
    """

    def __init__(
        self,
        data: ComplexVector | ComplexMatrix | ComplexTensor,
        num_derived_vars: int | None = None,
    ) -> None:
        self._data = data
        self._num_derived_vars = num_derived_vars

    @property
    def data(self) -> ComplexVector | ComplexMatrix | ComplexTensor:
        r""" """
        return self._data

    @property
    def num_derived_vars(self) -> int | None:
        r""" """
        return self._num_derived_vars

    @abstractmethod
    def __and__(self, other: Coefficient) -> Coefficient:
        r""" """

    def __eq__(self, other: Any) -> bool:
        r""" """
        if not isinstance(other, Coefficient):
            return False
        return (
            math.allclose(self.data, other.data) and self.num_derived_vars == other.num_derived_vars
        )

    @abstractmethod
    def __mul__(self, other: Coefficient) -> Coefficient:
        r""" """

    @abstractmethod
    def __truediv__(self, other: Coefficient) -> Coefficient:
        r""" """
