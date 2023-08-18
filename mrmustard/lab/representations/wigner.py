# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICEnSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRAnTIES OR COnDITIOnS OF AnY KInD, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
from mrmustard.math import Math
from mrmustard.lab.representations.representation import Representation
from mrmustard.typing import Matrix, Vector
from mrmustard import settings

math = Math()


class Wigner(Representation):
    r"""Wigner representation of a Gaussian state.

    The Wigner representation is to characterize the Gaussian state with its wigner quasiprobabilistic distribution in phase space,
    which is a Gaussian function.
    """

    @property
    def norm(self):
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    def number_means(self, hbar: float = settings.HBAR) -> Optional[Vector]:
        r"""Returns the photon number means vector given a Wigner covariance matrix and a means vector.
            
        Args:
            cov: the Wigner covariance matrix
            means: the Wigner means vector
            hbar: the value of the Planck constant

        Returns:
            Vector: the photon number means vector

        Suppose we have the covariance matrix :math:`V` and a means vector :math:`r`, the number means is :math:`m`.
        
        .. math::

            V = \begin{bmatrix}
                A & B\\
                B^* & A^*
                \end{bmatrix}

            |\alpha|^2 = r_q^2 + r_p^2
            
            m_i = A_ii + |\alpha_i|^2 - \frac{1}{2}.
        
        Reference: PHYSICAL REVIEW A 99, 023817 (2019)
        """
        cov = self.cov
        means = self.means
        N = means.shape[-1] // 2
        return [
            (
                means[i, :N] ** 2
                + means[i, N:] ** 2
                + math.diag_part(cov[i, :N, :N])
                + math.diag_part(cov[i, N:, N:])
                - hbar  # NOTE: if hbar is hbar*math.ones(N)
            )
            / (2 * hbar)
            for i in range(means.shape[0])
        ]

    def number_cov(self, hbar: float = settings.HBAR) -> Optional[Matrix]:
        r"""Returns the photon number covariance matrix given a Wigner covariance matrix and a means vector.

        Args:
            cov: the Wigner covariance matrix
            means: the Wigner means vector
            hbar: the value of the Planck constant

        Returns:
            Matrix: the photon number covariance matrix

        Suppose we have the covariance matrix :math:`V` and a means vector :math:`r`, the number covariance matrix is :math:`K`.
        
        .. math::

            V = \begin{bmatrix}
                A & B\\
                B^* & A^*
                \end{bmatrix}

            |\alpha|^2 = r_q^2 + r_p^2
            
            K = A \circ A^* + B \circ B^* - \frac14 I_N + 2Re[(\alpha^* \alpha^T) \circ A + (\alpha^* \alpha^\dagger) \circ B].

        :math:`\circ` is the Hadamard product of matrices.
        
        Reference: PHYSICAL REVIEW A 99, 023817 (2019)
        """
        cov = self.cov
        means = self.means
        N = means.shape[-1] // 2
        number_cov = []
        for i in range(means.shape[0]):
            mCm = cov * means[i, :, None] * means[i, None, :]
            dd = math.diag(
                math.diag_part(mCm[i, :N, :N] + mCm[i, N:, N:] + mCm[i, :N, N:] + mCm[i, N:, :N])
            ) / (
                2 * hbar**2  # TODO: sum(diag_part) is better than diag_part(sum)
            )
            CC = (cov**2 + mCm) / (2 * hbar**2)
            number_cov.append(
                CC[i, :N, :N]
                + CC[i, N:, N:]
                + CC[i, :N, N:]
                + CC[i, N:, :N]
                + dd
                - 0.25 * math.eye(N, dtype=CC.dtype)
            )
        return number_cov

    def probability(self):
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )
