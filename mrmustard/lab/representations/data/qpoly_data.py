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
#from numba import njit
import numpy as np
from typing import Tuple, Union
from mrmustard.math import Math
from mrmustard.lab import MatVecData, GaussianData
from mrmustard.typing import Batch, Matrix, Scalar, Vector

math = Math()
class QPolyData(MatVecData):

    def __init__(
        self,
        A: Batch[Matrix],
        b: Batch[Vector],
        c: Batch[Scalar],
    ) -> None:
        r"""
        Quadratic Gaussian data: quadratic coefficients, linear coefficients, constant.
        Each of these has a batch dimension, and the batch dimension is the same for all of them.
        They are the parameters of a Gaussian expressed as `c * exp(-x^T A x + x^T b)`.
        Args:
            A (batch, dim, dim): quadratic coefficients
            b (batch, dim): linear coefficients
            c (batch): constant
        """

        if isinstance(A, GaussianData):
            A, b, c = self._from_GaussianData(A=A)
        # TODO : make sure we're happy with this init. Nothing missing?

        super().__init__(mat=A, vec=b, coeff=c)



    def _from_GaussianData(self, A:GaussianData
                           ) -> Tuple[Batch[Matrix], Batch[Vector], Batch[Scalar]] :
        A = -math.inv(A.cov)
        b = math.inv(A.cov) @ A.mean
        c = A.coeff * np.einsum("bca,bcd,bde->bae", A.mean, math.inv(A.cov), A.mean)
        return A, b, c



    @property
    def A(self) -> Batch[Matrix]:
        return self.mat


    @A.setter
    def A(self, value) -> None:
        self.mat = value


    @property
    def b(self) -> Batch[Vector]:
        return self.vec


    @b.setter
    def b(self, value) -> None:
        self.vec = value


    @property
    def c(self) -> Batch[Scalar]:
        return self.coeff


    @c.setter
    def c(self, value) -> None:
        self.coeff = value



    def __truediv__(self, other:QPolyData) -> QPolyData:
       raise NotImplementedError() # TODO : implement!


    #@njit(parallel=True)
    def __mul__(self, other: Union[Scalar, QPolyData]) -> QPolyData:

        if type(other) is Scalar: # WARNING: this means we have to be very logical with our typing!
            c = super().scalar_mul(c=other)
            return self.__class__(A=self.A, b=self.b, c=c)
        
        else: # TODO : use MM's math module where possible
            raise NotImplementedError() # TODO : implement (is the below correct?)
            # try:
            #     return self.__class__(self.A + other.A, self.b + other.b, self.c * other.c)
            
            # except AttributeError:
            #     return self.__class__(self.A, self.b, self.c * other)

