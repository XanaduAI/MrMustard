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

from mrmustard.math import Math
from mrmustard.lab.representations.representation import Representation
from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.typing import Scalar, RealMatrix, RealVector, Matrix, Vector, Tensor
from typing import List
from mrmustard import settings


math = Math()

class Wigner(Representation):
    r""" Parent abstract class for the WignerKet and WignerDM representations.
    
    Args:
        cov: covariance matricx of the state (real symmetric) TODO: support only Gaussian state. If not Gaussian, cov can be complex.
        mean: mean vector of the state (real)
        coeffs: coefficients (complex) 
    """

    def __init__(self,
                 cov: Matrix, 
                 means: Vector, 
                 coeffs: Scalar = 1.0
                 ) -> None:

        self.data = MatVecData(cov=cov, means=means, coeffs=coeffs)
    

    @property
    def norm(self) -> float:
        #TODO: get the norm from other representation
        raise NotImplementedError()
    

    @property
    def number_means(self) -> RealVector:

        n = self.data.means.shape[-1] // 2

        cov_top_left = math.diag_part(self.data.cov[:n, :n])
        cov_bottom_right = math.diag_part(self.data.cov[n:, n:])
        covariance = cov_top_left + cov_bottom_right

        means_first_half = self.data.means[:n]
        means_second_half = self.data.means[n:]
        means = means_first_half **2 + means_second_half **2

        return (means + covariance - settings.HBAR) / (2 * settings.HBAR)
    

    #TODO : rename variables with actual names (apple, banana)
    @property
    def number_cov(self) -> RealMatrix:

        n = self.data.means.shape[-1] // 2

        extended_means_horizontal = self.data.means[:, None]
        extended_means_vertical = self.data.means[None, :]

        mCm = self.data.cov * extended_means_horizontal * extended_means_vertical

        # TODO: sum(diag_part) is better than diag_part(sum)
        diagonal = math.diag_part( mCm[:n, :n] + mCm[n:, n:] + mCm[:n, n:] + mCm[n:, :n] )
        diag_of_diag = math.diag( diagonal )

        CC = (self.data.cov**2 + mCm) / (2 * settings.HBAR**2)

        apple  = CC[:n, :n] + CC[n:, n:] + CC[:n, n:] + CC[n:, :n]

        banana = (0.25 * math.eye(n, dtype=CC.dtype))

        covariances = apple + (diag_of_diag / (2 * settings.HBAR**2)) - banana

        return covariances
    

    @property
    def number_variances(self) -> int:
        raise NotImplementedError()
    

    @property
    def probability(self) -> Tensor:
        raise NotImplementedError()