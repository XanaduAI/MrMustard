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
#from numba import njit
from typing import Optional, Union
from mrmustard.math import Math
from mrmustard.representations.data import MatVecData
from mrmustard.typing import Batched, Matrix, Scalar, Vector

math = Math()


class GaussianData(MatVecData):

    def __init__(
        self,
        cov: Optional[Batched[Matrix]] = None,
        mean: Optional[Batched[Vector]] = None,
        coeffs: Optional[Batched[Scalar]] = None,
    ) -> None:
        r"""
        Gaussian data: covariance, mean, coeffsicient.
        Each of these has a batch dimension, and the length of the
        batch dimension is the same for all three.
        These are the parameters of a linear combination of Gaussians,
        which is Gaussian if there is only one contribution for each.
        Each contribution parametrizes the Gaussian function:
        `coeffs * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.
        Args:
            cov (batch, dim, dim): covariance matrices (real symmetric)
            mean  (batch, dim): means (real)
            coeffs (batch): coeffsicients (complex)
        """
    
        if (cov or mean) is not None:
    
            if cov is None:
                dim = mean.shape[-1]
                batch_size = mean.shape[-2]
                cov = math.astensor([math.eye(dim, dtype=mean.dtype) for _ in range(batch_size)])

            else: # we know mean is None here
                dim = cov.shape[-1]
                batch_size = cov.shape[-3]
                mean = math.zeros((batch_size, dim), dtype=cov.dtype)
        else:
            raise ValueError("You need to define at one: covariance or mean")

        if coeffs is None:
            batch_size = cov.shape[-3]
            coeffs = math.ones((batch_size), dtype=mean.dtype)

        # what is this??? do we keep it ?
        if isinstance(cov, QuadraticPolyData):  # enables GaussianData(quadraticdata)
            poly = cov  # for readability
            inv_A = math.inv(poly.A)
            cov = 2 * inv_A
            mean = 2 * math.solve(poly.A, poly.b)
            coeffs = poly.c * math.cast(
                math.exp(0.5 * math.einsum("bca,bcd,bde->bae", mean, cov, mean)), poly.c.dtype
            )

        else: # why else, isn't this just part of the standard init???
            super().__init__(mat=cov, vec=mean, coeffs=coeffs)


        
    @property
    def cov(self) -> Batched[Matrix]:
        return self.mat



    @cov.setter
    def cov(self, value) -> None:
        self.mat = value



    @property
    def mean(self) -> Batched[Vector]:
        return self.vec



    @mean.setter
    def mean(self, value) -> None:
        self.vec = value

    

    def __truediv__(self, other: GaussianData) -> GaussianData:
       raise NotImplementedError() # TODO : implement!


    #@njit
    def __mul__(self, other: Union[Scalar, GaussianData]) -> GaussianData:

        if type(other) is Scalar: # WARNING: this means we have to be very logical with our typing!
            c = super().__scalar_mul(c=other)
            return self.__class__(cov=self.cov, mean=self.mean, coeffs=c)
        
        else:
            try:
                # covs = []
                # for c1 in self.cov:
                #     for c2 in other.cov:
                #         covs.append(math.matmul(c1, math.solve(c1 + c2, c2)))

                # means = []
                # for c1, m1 in zip(self.cov, self.mean):
                #     for c2, m2 in zip(other.cov, other.mean):
                #         means.append(
                #             math.matvec(c1, math.solve(c1 + c2, m2))
                #             + math.matvec(c2, math.solve(c1 + c2, m1))
                #         )

                # coeffs = []
                # for c1, m1, c2, m2, c3, m3, co1, co2 in zip(
                #     self.cov, self.mean, other.cov, other.mean, cov, mean, self.coeffs, other.coeffs
                # ):   
                #     coeffs.append(co1 * co2
                #         * math.exp(
                #             0.5 * math.sum(m1 * math.solve(c1, m1), axes=-1)
                #             + 0.5 * math.sum(m2 * math.solve(c2, m2), axes=-1)
                #             - 0.5 * math.sum(m3 * math.solve(c3, m3), axes=-1)
                #         )
                #     )

                covs = [math.matmul(c1, math.solve(c1 + c2, c2)) 
                                    for c1 in self.cov for c2 in other.cov]
                
                #means: c1 (c1 + c2)^-1 m2 + c2 (c1 + c2)^-1 m1 for each pair of cov mat in batch
                means = [ math.matvec(c1, math.solve(c1 + c2, m2)) 
                         + math.matvec(c2, math.solve(c1 + c2, m1))
                         for c1, m1 in zip(self.cov, self.mean)
                         for c2, m2 in zip(other.cov, other.mean)
                         ]

                cov = math.astensor(covs)
                mean = math.astensor(means)

                coeffs = [  co1 * co2
                            * math.exp(
                                0.5 * math.sum(m1 * math.solve(c1, m1), axes=-1)
                                + 0.5 * math.sum(m2 * math.solve(c2, m2), axes=-1)
                                - 0.5 * math.sum(m3 * math.solve(c3, m3), axes=-1)
                            )
                           for c1, m1, c2, m2, c3, m3, co1, co2 in zip(
                                                                    self.cov, self.mean, 
                                                                    other.cov, other.mean, 
                                                                    cov, mean, 
                                                                    self.coeffs, other.coeffs
                                                                    ) 
                        ]
                
                coeffs = math.astensor(coeffs)

                return self.__class__(cov=cov, mean=mean, coeffs=coeffs)

            except AttributeError as e:
                raise TypeError(f"Cannot tensor {self.__class__} and {other.__class__}.") from e
