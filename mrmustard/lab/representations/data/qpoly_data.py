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
import operator as op

from itertools import product
from typing import Optional, TYPE_CHECKING, Union

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, RealMatrix, Scalar, Vector

# if TYPE_CHECKING: # This is to avoid the circular import issu with GaussianData<>QPolyData
#     from mrmustard.lab.representations.data.gaussian_data import GaussianData


math = Math()


class QPolyData(MatVecData):
    r"""Quadratic polynomial data for certain Representation objects.

    Quadratic Gaussian data is made of: quadratic coefficients, linear coefficients, constant.
    Each of these has a batch dimension, and the batch dimension is the same for all of them.
    They are the parameters of a Gaussian expressed as `c * exp(-x^T A x + x^T b)`.

    Note that if constants are not provided, they will all be initialized at 1.

    Args:
        mat:    series of quadratic coefficient
        vec:    series of linear coefficients
        c:      series of constants
    """

    def __init__(self, A: Batch[RealMatrix], b: Batch[Vector], c: Optional[Batch[Scalar]]) -> None:
        if c is None: #default cs should all be 1
            n = b.shape[0] # number of elements
            c = np.repeat(1.0, n)

        if self.helper_check_is_real_symmetric(A):
            super().__init__(mat=A, vec=b, coeffs=c)

        else:
            raise ValueError("Matrix A is not real symmetric, object can't be initialized.")

    @property
    def A(self) -> Batch[RealMatrix]:
        return self.mat

    @property
    def b(self) -> Batch[Vector]:
        return self.vec

    @property
    def c(self) -> Batch[Scalar]:
        return self.coeffs

    def __mul__(self, other: Union[Scalar, QPolyData]) -> QPolyData:
        if isinstance(other, QPolyData):  # TODO: proof it against other objects
            new_a = self._operate_all_combinations(self.A, other.A, op.add)
            new_b = self._operate_all_combinations(self.b, other.b, op.add)
            new_coeffs = self._operate_all_combinations(self.c, other.c, op.mul)
            return self.__class__(A=new_a, b=new_b, c=new_coeffs)
        else:
            try:  # scalar
                new_coeffs = np.fromiter(map(lambda x: x * other ,self.c), dtype=np.float64)
                return self.__class__(self.A, self.b, new_coeffs)
            except TypeError as e:  # Neither same object type nor a scalar case
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
            
    def _operate_all_combinations(self, X, Y, operator):
        """Returns the element-wise operation on the cartesian product of inputs X and Y."""
        both = product(X,Y)
        return np.fromiter(map(lambda z: operator(z[0], z[1]), both), dtype=np.float64)