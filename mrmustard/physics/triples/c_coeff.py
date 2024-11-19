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

from mrmustard import math

from .base import Coefficient

__all__ = ["cCoeff"]


class cCoeff(Coefficient):
    r""" """

    def __and__(self, other: cCoeff) -> cCoeff:
        r""" """
        return self * other

    def __mul__(self, other: cCoeff) -> cCoeff:
        r""" """
        c1 = self.data
        c2 = other.data
        c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
        return cCoeff(c3)

    def __truediv__(self, other: cCoeff) -> cCoeff:
        r""" """
        c1 = self.data
        c2 = 1 / other.data
        c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
        return cCoeff(c3)
