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

from typing import Optional, TYPE_CHECKING, Union

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, Matrix, Scalar, Tensor, Vector


if TYPE_CHECKING:  # This is to avoid the circular import issue with GaussianData<>QPolyData
    from mrmustard.lab.representations.data.qpoly_data import QPolyData

math = Math()


class GaussianData(MatVecData):
    r"""Gaussian data for certain representation objects.

    Gaussian data is made of covariance, mean vector and coefficient. Each of these has a batch
    dimension, and the length of the batch dimension is the same for all three.
    These are the parameters of a linear combination of Gaussians, which is Gaussian if there is
    only one contribution for each.
    Each contribution parametrizes the Gaussian function:
    `coeffs * exp(-0.5*(x-mean)^T cov^-1 (x-mean))/sqrt((2pi)^k det(cov))` where k is the size of cov.

    Args:
        cov (Optional[Batch[Matrix]]):      covariance matrices (real symmetric)
        means (Optional[Batch[Vector]]):    mean vector of the state (real), note that the
                                            dimension must be even
        coeffs (Optional[Batch[Scalar]]):   coefficients (complex)

    Raises:
        ValueError: if neither means nor covariance is defined
    """

    def __init__(
        self,
        cov: Optional[Batch[Matrix]] = None,
        means: Optional[Batch[Vector]] = None,
        coeffs: Optional[Batch[Scalar]] = None,
    ) -> None:
        if cov is not None or means is not None:  # at least one is defined -or both-
            if cov is None:
                dim = means.shape[-1]
                batch_size = means.shape[0]
                cov = math.astensor([math.eye(dim, dtype=means.dtype) for _ in range(batch_size)])

            elif means is None:  # we know cov is not None here
                dim = cov.shape[-1]
                batch_size = cov.shape[0]
                means = math.zeros((batch_size, dim), dtype=cov.dtype)
        else:
            raise ValueError("You need to define at last one of covariance or mean")

        # self.num_modes = means.shape[-1] // 2
        super().__init__(mat=cov, vec=means, coeffs=coeffs)

    @property
    def cov(self) -> Matrix:
        return self.mat

    @property
    def means(self) -> Vector:
        return self.vec

    @property
    def c(self) -> Scalar:
        return self.coeffs

    def value(self, x: np.array):
        r"""returns the value of the gaussian at x.

        Arguments:
            x (array of floats): where to evaluate the function
        """
        val = 0.0
        for sigma, mu, c in zip(self.cov, self.means, self.c):
            exponent = -0.5 * math.sum(math.solve(sigma, (x - mu)) * (x - mu))
            denom = math.sqrt((2 * np.pi) ** len(x) * math.det(sigma))
            val += c * math.exp(exponent) / denom
        return val

    def __mul__(self, other: Union[Scalar, GaussianData]) -> GaussianData:
        if isinstance(other, GaussianData):
            joint_covs = self._compute_mul_covs(other=other)
            joint_means = self._compute_mul_means(other=other)
            joint_coeffs = self._compute_mul_coeffs(other=other)
            return self.__class__(cov=joint_covs, means=joint_means, coeffs=joint_coeffs)
        else:
            try:  # hope it's a scalar
                new_coeffs = self.coeffs * other
                return self.__class__(cov=self.cov, means=self.means, coeffs=new_coeffs)
            except TypeError as e:  # Neither GaussianData nor scalar
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def _compute_mul_covs(self, other: GaussianData) -> Tensor:
        r"""Computes the combined covariances when multiplying Gaussians.
        The formula is cov1 (cov1 + cov2)^-1 cov2 for each pair of cov1 and cov2
        from self and other (see https://math.stackexchange.com/q/964103)

        Args:
            other (GaussianData): another GaussianData object which covariance will be multiplied

        Returns:
            (Tensor) The tensor of combined covariances
        """
        combined_covs = [
            math.matmul(c1, math.solve(c1 + c2, c2)) for c1 in self.cov for c2 in other.cov
        ]
        return math.astensor(combined_covs)

    def _compute_mul_means(self, other: GaussianData) -> Tensor:
        r"""Computes the combined means when multiplying Gaussians.
        The formula is cov1 (cov1 + cov2)^-1 mu2 + cov2 (cov1 + cov2)^-1 mu1 for each
        pair of (cov1, mu1) and (cov2, mu2) from self and other.
        (see https://math.stackexchange.com/q/964103)

        Args:
            other (GaussianData):  another GaussianData object which means will be multiplied

        Returns:
            (Tensor) The tensor of combined multiplied means
        """
        combined_means = [
            math.matvec(c1, math.solve(c1 + c2, m2)) + math.matvec(c2, math.solve(c1 + c2, m1))
            for c1, m1 in zip(self.cov, self.means)
            for c2, m2 in zip(other.cov, other.means)
        ]
        return math.astensor(combined_means)

    def _compute_mul_coeffs(self, other: GaussianData) -> Tensor:
        r"""Computes the combined coefficients when multiplying Gaussians.

        Args:
            other (GaussianData):   another GaussianData object which coeffs will be multiplied

        Returns:
            (Tensor) The tensor of multiplied coefficients
        """
        combined_coeffs = [
            c1 * c2 * self.__class__(cov=[cov1 + cov2], means=[m1], coeffs=[1]).value(m2)
            for cov1, m1, c1 in zip(self.cov, self.means, self.c)
            for cov2, m2, c2 in zip(other.cov, other.means, other.c)
        ]
        return math.astensor(combined_coeffs)
