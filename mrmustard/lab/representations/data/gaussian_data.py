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

from typing import Optional, TYPE_CHECKING, Union

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Matrix, Scalar, Tensor, Vector


if TYPE_CHECKING:  # This is to avoid the circular import issu with GaussianData<>QPolyData
    from mrmustard.lab.representations.data.qpoly_data import QPolyData

math = Math()


class GaussianData(MatVecData):
    r"""Gaussian data for certain representation objects.

    Gaussian data is made of covariance, mean vector and coefficient. Each of these has a batch
    dimension, and the length of the batch dimension is the same for all three.
    These are the parameters of a linear combination of Gaussians, which is Gaussian if there is
    only one contribution for each.
    Each contribution parametrizes the Gaussian function:
    `coeffs * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.

    Args:
        cov:    covariance matrices (real symmetric)
        means:  mean vector of the state (real), note that the dimension must be even
        coeffs: coefficients (complex)

    Raises:
        ValueError: if neither means nor covariance is defined
    """

    def __init__(
        self,
        cov: Optional[Matrix] = None,
        means: Optional[Vector] = None,
        coeffs: Optional[Scalar] = 1.0,
    ) -> None:
        # TODO: BATCH
        if cov is not None or means is not None:  # at least one is defined -or both-
            if cov is None:
                self.num_modes = means.shape[0] // 2
                cov = math.eye(2 * self.num_modes, dtype=means.dtype)
                # batch_size = mean.shape[-2]
                # cov = math.astensor( list( repeat( math.eye(dim, dtype=mean.dtype), batch_size )))

            elif means is None:  # we know cov is not None here
                self.num_modes = cov.shape[-1] // 2
                means = math.zeros(2 * self.num_modes, dtype=cov.dtype)

                # batch_size = cov.shape[-3]
                # mean = math.zeros( (batch_size, dim), dtype=cov.dtype )
        else:
            raise ValueError("You need to define at one: covariance or mean")

        # if coeffs is None:
        #     coeffs = 1.0
        # batch_size = cov.shape[-3]
        # coeffs = math.ones((batch_size), dtype=mean.dtype)

        # if isinstance(cov, QPolyData): #NOTE: what do we do about this? support or not?
        #     cov, mean, coeffs = self._from_QPolyData(poly=cov)
        # Robertson–Schr ̈odinger uncertainty relation for a (Gaussian) quantum state
        # if (cov + 1j*sympmat(self.num_modes)).numpy().all() >= 0:
        #     raise ValueError("The covariance matrix is not valid. cov + i\Omega < 0.")
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

    def __mul__(self, other: Union[Scalar, GaussianData]) -> GaussianData:
        try:
            joint_covs = self._compute_mul_covs(other=other)

            joint_means = self._compute_mul_means(other=other)

            joint_coeffs = self._compute_mul_coeffs(
                other=other, joint_covs=joint_covs, joint_means=joint_means
            )
            return self.__class__(cov=joint_covs, means=joint_means, coeffs=joint_coeffs)
        except AttributeError:
            new_coeffs = self.coeffs * other
            return self.__class__(cov=self.cov, means=self.means, coeffs=new_coeffs)
        except TypeError as e:
            raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def _compute_mul_covs(self, other: GaussianData) -> Tensor:
        r"""Computes the combined covariances when multiplying Gaussian-represented states.

        Args:
            other: another GaussianData object which covariance will be multiplied

        Returns:
            The tensor of combined covariances
        """
        combined_covs = [
            math.matmul(c1, math.solve(c1 + c2, c2)) for c1 in self.cov for c2 in other.cov
        ]
        return math.astensor(combined_covs)

    def _compute_mul_coeffs(
        self, other: GaussianData, joint_covs: Tensor, joint_means: Tensor
    ) -> Tensor:
        r"""Computes the combined coefficients when multiplying Gaussian-represented states.

        Args:
            other:          another GaussianData object which coeffs will be multiplied
            joint_covs:     the combined covariances of the two objects
            joint_means:    the combined means of the two objects

        Returns:
            The tensor of multiplied coefficients
        """
        combined_coeffs = [
            co1
            * co2
            * math.exp(
                0.5 * math.sum(m1 * math.solve(c1, m1), axes=-1)
                + 0.5 * math.sum(m2 * math.solve(c2, m2), axes=-1)
                - 0.5 * math.sum(m3 * math.solve(c3, m3), axes=-1)
            )
            for c1, m1, c2, m2, c3, m3, co1, co2 in zip(
                self.cov,
                self.means,
                other.cov,
                other.means,
                joint_covs,
                joint_means,
                self.coeffs,
                other.coeffs,
            )
        ]
        return math.astensor(combined_coeffs)

    def _compute_mul_means(self, other: GaussianData) -> Tensor:
        r"""Computes the combined means when multiplying Gaussian-represented states.

        Formula correpsonds to : c1 (c1 + c2)^-1 m2 + c2 (c1 + c2)^-1 m1 for each pair of cov mat
        in batch.

        Args:
            other:  another GaussianData object which means will be multiplied

        Returns:
            The tensor of combined multiplied means
        """
        combined_means = [
            math.matvec(c1, math.solve(c1 + c2, m2)) + math.matvec(c2, math.solve(c1 + c2, m1))
            for c1, m1 in zip(self.cov, self.means)
            for c2, m2 in zip(other.cov, other.means)
        ]
        return math.astensor(combined_means)

    # @staticmethod
    # def _from_QPolyData(poly:QPolyData
    #                     ) -> Tuple[Batch[Matrix], Batch[Vector], Batch[Scalar]] :
    #     r""" Extracts necessary information from a QPolyData object to build a GaussianData one.

    #     Args:
    #         poly: the quadratic polynomial data

    #     Returns:
    #         The necessary matrix vector and coefficients to build a GaussianData object
    #     """
    #     inv_A = math.inv(poly.A)
    #     cov = 2 * inv_A

    #     mean = 2 * math.solve(poly.A, poly.b)

    #     pre_coeffs = math.cast( math.exp( 0.5 * math.einsum("bca,bcd,bde->bae", mean, cov, mean)),
    #                             dtype=poly.c.dtype
    #                             )
    #     coeffs = poly.c * pre_coeffs

    #     return (cov, mean, coeffs)

    # old code for mul TODO: do we keep?
    # def __mul__(self, other: Union[Scalar, GaussianData]) -> GaussianData:
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
