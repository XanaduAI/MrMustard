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
from mrmustard.lab.representations.data.gaussian_data import GaussianData
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

        self.data = GaussianData(cov=cov, means=means, coeffs=coeffs)
        
    
    @property
    def purity(self) -> float:
        return 1 / math.sqrt(math.det((2 / settings.HBAR) * self.data.cov))
    

    @property
    def norm(self) -> float:
        #TODO: get the norm from other representation
        raise NotImplementedError()
    

    @property
    def von_neumann_entropy(self) -> float:
        symplectic_eigvals = self.symplectic_eigenvals(self.data.cov, settings.HBAR)
        entropy = math.sum(self._g(symplectic_eigvals))
        return entropy
    

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
    

    @property
    def symplectic_eigenvals(self) -> List[Scalar]:
        r""" Computes the sympletic eigenspectrum of a covariance matrix.

        Note that for a pure state, we expect the sympletic eigenvalues to be 1.

        Returns:
            The sympletic (even) eigenvalues
        """

        J = math.J(self.data.cov.shape[-1] // 2)  # create a sympletic form
        M = math.matmul(1j * J, self.data.cov * (2 / settings.HBAR))

        eigenspectrum = math.eigvals(M)

        return math.abs(eigenspectrum[::2]) # TODO: sort?


    # def trace(self, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]: #NOTE: move to physics after MVP
    #     r""" Returns the covariances and means after discarding the specified modes.

    #     Args:
    #         Bmodes: modes to discard

    #     Returns:
    #         The covariance matrix and the means vector after discarding the specified modes
    #     """
    #     n = len(self.data.cov) // 2

    #     good_modes, good_modes_plus_n = self._yield_correct_modes(n=n, bad_modes=Bmodes)

    #     Aindices = math.astensor(good_modes + good_modes_plus_n)

    #     A_cov_block = math.gather(math.gather(self.data.cov, Aindices, axis=0), Aindices, axis=1)
    #     A_means_vec = math.gather(self.data.means, Aindices)

    #     return A_cov_block, A_means_vec   


    # def partition_cov(self, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]: #NOTE: move to physics after MVP
    #     r""" Partitions the covariance matrix into the ``A`` and ``B`` subsystems and the AB 
    #     coherence block.

    #     Args:
    #         Amodes: the modes of system ``A``

    #     Returns:
    #         Tuple[Matrix, Matrix, Matrix]: the cov of ``A``, the cov of ``B`` and the AB block
    #     """
    #     n = self.data.cov.shape[-1] // 2

    #     good_modes, good_modes_plus_n = self._yield_correct_modes(n=n, bad_modes=Amodes)

    #     Bindices = math.cast(good_modes + good_modes_plus_n, "int32")
    #     B_block = math.gather(math.gather(self.data.cov, Bindices, axis=1), Bindices, axis=0)

    #     amodes_plus_n = self._add_element_wise_n(n=n, l=Amodes)
    #     Aindices = math.cast(Amodes + amodes_plus_n, "int32")
    #     A_block = math.gather(math.gather(self.data.cov, Aindices, axis=1), Aindices, axis=0)
        
    #     AB_block = math.gather(math.gather(self.data.cov, Bindices, axis=1), Aindices, axis=0)

    #     return A_block, B_block, AB_block


    # def partition_means(self, Amodes: Sequence[int]) -> Tuple[Vector, Vector]: #NOTE: move to physics after MVP
    #     r"""Partitions the means vector into the ``A`` and ``B`` subsystems.

    #     Args:
    #         Amodes (Sequence[int]): the modes of system ``A``

    #     Returns:
    #         Tuple[Vector, Vector]: the means of ``A`` and the means of ``B``
    #     """
    #     n = len(self.data.means) // 2

    #     good_modes, good_modes_plus_n = self._yield_correct_modes(n=n, bad_modes=Amodes)

    #     Bindices = math.cast(good_modes + good_modes_plus_n,"int32")

    #     amodes_plus_n = self._add_element_wise_n(n=n, l=Amodes)
    #     Aindices = math.cast(Amodes + amodes_plus_n, "int32")

    #     return math.gather(self.data.means, Aindices), math.gather(self.data.means, Bindices)
    

    # @staticmethod
    # def _g(x:List[Scalar]) -> List[Scalar]:  #NOTE: move to physics after MVP
    #     r""" Used exclusively to compute the Wigner Von neumann entropy.

    #     Args:
    #         x: the symplectic eigenvalues 

    #     References:


    #     Returns:
                    
    #     """
    #     return math.xlogy((x + 1) / 2, (x + 1) / 2) - math.xlogy((x - 1) / 2, (x - 1) / 2 + 1e-9)
    

    # def _yield_correct_modes(self, n:int, bad_modes:Sequence[int]) -> Tuple(List[int]): #NOTE: move to physics after MVP
    #     r""" Helper function to select only desired modes based on a list of undesired ones.

    #     Args:
    #         n: the total range of the number of modes
    #         bad_modes: the modes we wish to discard

    #     Returns:
    #         A tuple containing the list of desired modes on the left and the list of desired modes 
    #         plus n on the right.
    #     """
    #     good_modes = list(set(range(n)).difference(bad_modes))
    #     good_modes_plus_n =  self._add_element_wise_n(n=n, l=good_modes)
    #     return (good_modes, good_modes_plus_n)
    

    # @staticmethod
    # def _add_element_wise_n(n:int, l:List[int]) -> List[int]: #NOTE: move to physics after MVP
    #     r""" Helper function to map a +n to all elements. """
    #     return list(map( lambda x: x+n , l))

