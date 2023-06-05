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

from mrmustard.representations.data import MatVecData, GaussianData

class QPolyData(MatVecData):

    def __init__(
        self,
        A: Batched[Matrix],
        b: Batched[Vector],
        c: Batched[Scalar],
    ) -> QPolyData:
        r"""
        Quadratic Gaussian data: quadratic coefficients, linear coefficients, constant.
        Each of these has a batch dimension, and the batch dimension is the same for all of them.
        They are the parameters of a Gaussian expressed as `c * exp(-x^T A x + x^T b)`.
        Args:
            A (batch, dim, dim): quadratic coefficients
            b (batch, dim): linear coefficients
            c (batch): constant
        """
        if isinstance(A, GaussianData): # isn't there a scope problem here ???
                A = -math.inv(A.cov)
                b = math.inv(A.cov) @ A.mean
                c = A.coeff * np.einsum("bca,bcd,bde->bae", A.mean, math.inv(A.cov), A.mean)

        super().__init__(mat=A, vec=b, coeff=c)



    @property
    def A(self) -> Batched[Matrix]:
        return self.mat



    @A.setter
    def A(self, value):
        self.mat = value



    @property
    def b(self) -> Batched[Vector]:
        return self.vec



    @b.setter
    def b(self, value):
        self.vec = value



    @property # isn't it confusing to have c then coeff? why not just coeff and leave it at that ???
    def c(self) -> Batched[Scalar]:
        return self.coeff



    @c.setter
    def c(self, value):
        self.coeff = value



    def __truediv__():
       raise NotImplementedError() # TODO : implement!



    def __mul__(self, other: Union[Number, QuadraticPolyData]) -> QuadraticPolyData:

        if self.__class__ != other.__class__ and type(other) != Number:
            raise TypeError(f"Cannot multiply GaussianData with {other.__class__.__qualname__}")
            # TODO: change the error? not sure the way it's written supports anything... qualname?
        
        else:
            try:
                return QuadraticPolyData(self.A + other.A, self.b + other.b, self.c * other.c)
            
            except AttributeError:
                return QuadraticPolyData(self.A, self.b, self.c * other)

