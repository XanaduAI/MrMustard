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

from mrmustard.math import Math
from mrmustard.lab.representations import Representation
from mrmustard.lab.representations.data import GaussianData
from mrmustard.typing import Scalar, RealMatrix, RealVector, Matrix, Vector, Tuple
from typing import Sequence

math = Math()

class Wigner(Representation):

    def __init__(self, cov, means):
        super().__init__()
        self.data = GaussianData(cov, means)


    def purity(cov: RealMatrix, hbar: float) -> Scalar:
        r"""Returns the purity of the state with the given covariance matrix.

        Args:
            cov (Matrix): the covariance matrix
            hbat (float) ; the reduced Plank constant

        Returns:
            float: the purity
        """
        return 1 / math.sqrt(math.det((2 / hbar) * cov))
    

    def number_means(cov: RealMatrix, means: RealVector, hbar: float) -> RealVector:
        r"""Returns the photon number means vector given a Wigner covariance matrix and a means vector.

        Args:
            cov: the Wigner covariance matrix
            means: the Wigner means vector
            hbar: the value of the Planck constant

        Returns:
            Vector: the photon number means vector
        """
        N = means.shape[-1] // 2
        return (
            means[:N] ** 2
            + means[N:] ** 2
            + math.diag_part(cov[:N, :N])
            + math.diag_part(cov[N:, N:])
            - hbar
        ) / (2 * hbar)
    

    def number_cov(cov: RealMatrix, means: RealVector, hbar: float) -> RealMatrix:
        r"""Returns the photon number covariance matrix given a Wigner covariance matrix and a means vector.

        Args:
            cov: the Wigner covariance matrix
            means: the Wigner means vector
            hbar: the value of the Planck constant

        Returns:
            Matrix: the photon number covariance matrix
        """
        N = means.shape[-1] // 2
        mCm = cov * means[:, None] * means[None, :]
        dd = math.diag(math.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (
            2 * hbar**2  # TODO: sum(diag_part) is better than diag_part(sum)
        )
        CC = (cov**2 + mCm) / (2 * hbar**2)
        return (
            CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * math.eye(N, dtype=CC.dtype)
        )
    
    
    def symplectic_eigenvals(cov: RealMatrix, hbar: float) -> list:
        r"""Returns the sympletic eigenspectrum of a covariance matrix.

        For a pure state, we expect the sympletic eigenvalues to be 1.

        Args:
            cov (Matrix): the covariance matrix
            hbar (float): the value of the Planck constant

        Returns:
            List[float]: the sympletic eigenvalues
        """
        J = math.J(cov.shape[-1] // 2)  # create a sympletic form
        M = math.matmul(1j * J, cov * (2 / hbar))
        vals = math.eigvals(M)  # compute the eigenspectrum
        return math.abs(vals[::2])  # return the even eigenvalues  # TODO: sort?


    def von_neumann_entropy(cov: RealMatrix, hbar: float) -> float:
        r"""Returns the Von Neumann entropy.

        For a pure state, we expect the Von Neumann entropy to be 0.

        Reference: (https://arxiv.org/pdf/1110.3234.pdf), Equations 46-47.

        Args:
            cov (Matrix): the covariance matrix

        Returns:
            float: the Von Neumann entropy
        """

        def g(x):
            return math.xlogy((x + 1) / 2, (x + 1) / 2) - math.xlogy((x - 1) / 2, (x - 1) / 2 + 1e-9)

        symp_vals = symplectic_eigenvals(cov, hbar)
        entropy = math.sum(g(symp_vals))
        return entropy
    

    def trace(cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
        r"""Returns the covariances and means after discarding the specified modes.

        Args:
            cov (Matrix): covariance matrix
            means (Vector): means vector
            Bmodes (Sequence[int]): modes to discard

        Returns:
            Tuple[Matrix, Vector]: the covariance matrix and the means vector after discarding the specified modes
        """
        N = len(cov) // 2
        Aindices = math.astensor(
            [i for i in range(N) if i not in Bmodes] + [i + N for i in range(N) if i not in Bmodes]
        )
        A_cov_block = math.gather(math.gather(cov, Aindices, axis=0), Aindices, axis=1)
        A_means_vec = math.gather(means, Aindices)
        return A_cov_block, A_means_vec   


    def partition_cov(cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
        r"""Partitions the covariance matrix into the ``A`` and ``B`` subsystems and the AB coherence block.

        Args:
            cov (Matrix): the covariance matrix
            Amodes (Sequence[int]): the modes of system ``A``

        Returns:
            Tuple[Matrix, Matrix, Matrix]: the cov of ``A``, the cov of ``B`` and the AB block
        """
        N = cov.shape[-1] // 2
        Bindices = math.cast(
            [i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes],
            "int32",
        )
        Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
        A_block = math.gather(math.gather(cov, Aindices, axis=1), Aindices, axis=0)
        B_block = math.gather(math.gather(cov, Bindices, axis=1), Bindices, axis=0)
        AB_block = math.gather(math.gather(cov, Bindices, axis=1), Aindices, axis=0)
        return A_block, B_block, AB_block


    def partition_means(means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
        r"""Partitions the means vector into the ``A`` and ``B`` subsystems.

        Args:
            means (Vector): the means vector
            Amodes (Sequence[int]): the modes of system ``A``

        Returns:
            Tuple[Vector, Vector]: the means of ``A`` and the means of ``B``
        """
        N = len(means) // 2
        Bindices = math.cast(
            [i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes],
            "int32",
        )
        Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
        return math.gather(means, Aindices), math.gather(means, Bindices)