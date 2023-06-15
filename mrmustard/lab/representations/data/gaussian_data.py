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
from typing import Optional, Tuple, Union, TYPE_CHECKING
from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, Matrix, Scalar, Vector

if TYPE_CHECKING: # This is to avoid the circular import issu with GaussianData<>QPolyData
    from mrmustard.lab.representations.data import QPolyData

math = Math()

class GaussianData(MatVecData):
    r""" Gaussian data for certain representation objects.

    Gaussian data is made of covariance, mean and coefficient. Each of these has a batch dimension, 
    and the length of the batch dimension is the same for all three.
    These are the parameters of a linear combination of Gaussians, which is Gaussian if there is 
    only one contribution for each.
    Each contribution parametrizes the Gaussian function:
    `coeffs * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.

    Args:
        cov: covariance matrices (real symmetric)
        mean: means (real)
        coeffs: coefficients (complex)
    """

    def __init__(self,
        cov: Optional[Batch[Matrix]] = None,
        mean: Optional[Batch[Vector]] = None,
        coeffs: Optional[Batch[Scalar]] = None
        ) -> None:
        # Done here because of circular import with GaussianData<>QPolyData
        from mrmustard.lab.representations.data import QPolyData
    
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

        if isinstance(cov, QPolyData):
            cov, mean, coeffs = self._from_QPolyData(poly=cov)

        super().__init__(mat=cov, vec=mean, coeffs=coeffs)


    @property
    def cov(self) -> Batch[Matrix]:
        return self.mat

    @cov.setter
    def cov(self, value) -> None:
        self.mat = value

    @property
    def mean(self) -> Batch[Vector]:
        return self.vec

    @mean.setter
    def mean(self, value) -> None:
        self.vec = value

    
    @staticmethod
    def _from_QPolyData(poly:QPolyData
                        ) -> Tuple[Batch[Matrix], Batch[Vector], Batch[Scalar]] :
        r""" Extracts necessary information from a QPolyData object to build a GaussianData one.

        Args:
            poly: the quadratic polynomial data

        Returns:
            The necessary matrix vector and coefficients to build a GaussianData object
        """ 
        inv_A = math.inv(poly.A)
        cov = 2 * inv_A
        mean = 2 * math.solve(poly.A, poly.b)
        coeffs = poly.c * math.cast(
            math.exp(0.5 * math.einsum("bca,bcd,bde->bae", mean, cov, mean)), poly.c.dtype
        )

        return (cov, mean, coeffs)


    def __truediv__(self, other: Scalar) -> GaussianData:
        return self.__class__(cov=self.cov, mean=self.mean, coeffs=self.coeffs/other)


    def __mul__(self, other: Union[Scalar, GaussianData]) -> GaussianData:
        if isinstance(other, Scalar):
            return self.__class__(cov=self.cov, mean=self.mean, coeffs=self.coeffs*other)
        
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
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
