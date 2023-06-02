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

from mrmustard.representations.data import MatVecData
from typing import Optional, Union
from mrmustard.types import Batched, Matrix, Scalar, Vector

math = Math()

class GaussianData(MatVecData):

    def __init__(
        self,
        cov: Optional[Batched[Matrix]] = None,
        mean: Optional[Batched[Vector]] = None,
        coeff: Optional[Batched[Scalar]] = None,
    ) -> GaussianData: # wth is wrong with variable scope???
        r"""
        Gaussian data: covariance, mean, coefficient.
        Each of these has a batch dimension, and the length of the
        batch dimension is the same for all three.
        These are the parameters of a linear combination of Gaussians,
        which is Gaussian if there is only one contribution for each.
        Each contribution parametrizes the Gaussian function:
        `coeff * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.
        Args:
            cov (batch, dim, dim): covariance matrices (real symmetric)
            mean  (batch, dim): means (real)
            coeff (batch): coefficients (complex)
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

        if coeff is None:
            batch_size = cov.shape[-3]
            coeff = math.ones((batch_size), dtype=mean.dtype)

        if isinstance(cov, QuadraticPolyData):  # enables GaussianData(quadraticdata) do we keep this???
            poly = cov  # for readability
            inv_A = math.inv(poly.A)
            cov = 2 * inv_A
            mean = 2 * math.solve(poly.A, poly.b)
            coeff = poly.c * math.cast(
                math.exp(0.5 * math.einsum("bca,bcd,bde->bae", mean, cov, mean)), poly.c.dtype
            )

        else: # why else???
            super().__init__(cov, mean, coeff)