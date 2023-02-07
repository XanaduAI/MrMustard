# Copyright 2022 Xanadu Quantum Technologies Inc.

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

from abc import ABC, abstractmethod

from mrmustard import settings
from mrmustard.math import Math

math = Math()


class Data(ABC):
    r"""Data is a class that holds the information that is necessary to define a
    state/operation/measurement in any representation. It enables algebraic operations
    for all types of data (gaussian, array, samples, symbolic, etc).
    Algebraic operations act on the Hilbert space vectors or on ensembles
    of Hilbert vectors (which form a convex structure via the Giri monad).
    The sopecific implementation depends on both the representation and the kind of data used by it.

    These are the supported data types (as either Hilbert vectors or convex
    combinations of Hilbert vectors):
    - Gaussian generally stacks cov, mean, coeff along a batch dimension.
    - QuadraticPoly is a representation of the Gaussian as a quadratic polynomial.
    - Array (ket/dm) operates with the data arrays themselves.
    - Samples (samples) operates on the (x,f(x)) pairs with interpolation.
    - Symbolic (symbolic) operates on the symbolic expression via sympy.

    This class is abstract and Gaussian, Array, Samples, Symbolic inherit from it.
    """

    @property
    def preferred(self):
        for data_type in settings.PREFERRED_DATA_ORDER:
            if hasattr(self, data_type):
                return getattr(self, data_type)

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __and__(self, other):  # tensor product
        pass

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __neg__(self):
        return -1 * self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)  # this numerically naughty
