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
from typing import Optional
from thewalrus.symplectic import is_symplectic
from strawberryfields.decompositions import williamson
import numpy as np
from mrmustard import settings
from mrmustard.lab.representations.wigner import Wigner
from mrmustard.lab.representations.data.symplectic_data import SymplecticData
from mrmustard.typing import Matrix, Vector, Scalar
from mrmustard.math import Math

math = Math()


class WignerKet(Wigner):
    r"""The Wigner ket representation is to characterize the pure Gaussian state with its wigner quasiprobabilistic distribution in phase space,
    which is a Gaussian function. This Gaussian function is characterized by a mean vector and a covariance matrix.

    WignerKet class is a special class since the evolution of the pure state under unitary operators can be described by the multiplication
    of their symplectic matrices. So that in this case, we need only store the symplectic matrix is enough for the pure Gaussian state and its
    covaraince matrix can be obtained from its symplectic matrix :math:`\frac{\hbar}{2}SS^T`.

    Args:
        symplectic (Optional[Matrix]): symplectic matrices and the first dimension is the batch dimension indicates the linear combination of different WignerKet Classes.
        displacement (Optional[Vector]): dispalcement vectors and the first dimension is the batch dimension indicates the linear combination of different WignerKet Classes.
        coeffs (Optional[Scalar]): coefficients of the state and the length of is is the batch dimensionthe first dimension is the batch dimension indicates the linear combination of different WignerKet Classes.

    Properties:
        cov: the covariance matrix calculating from its symplectic matrix.
        means: the same as the displacement vector.
    """

    def __init__(
        self,
        symplectic: Optional[Matrix],
        displacement: Optional[Vector],
        coeffs: Optional[Scalar] = 1.0,
    ) -> None:
        # Check the symplecticity of the matrix
        def check_symplectic(sym):
            if not is_symplectic(sym):
                raise ValueError("The matrix is not symplectic!")

        if symplectic.shape == 3:
            for i in range(symplectic.shape[0]):
                check_symplectic(symplectic[i, :])
        elif symplectic.shape == 2:
            check_symplectic(symplectic)
            symplectic = math.expand_dims(symplectic, axis=0)
            displacement = math.expand_dims(displacement, axis=0)

        self.data = SymplecticData(symplectic=symplectic, displacement=displacement, coeffs=coeffs)

    @property
    def cov(self) -> Optional[Matrix]:
        "Returns the covariance matrix/matrices of the state."
        return [
            (
                settings.HBAR
                / 2
                * math.matmul(self.data.symplectic[i], math.transpose(self.data.symplectic[i]))
            )
            for i in self.data.symplectic.shape[0]
        ]

    @property
    def means(self) -> Optional[Vector]:
        "Returns the means vector(s) of the state."
        return self.data.displacement

    @property
    def purity(self) -> float:
        return 1.0

    @classmethod
    def from_covariance(cls, cov, means):
        r"""This function allows us to construct a WignerKet class state from a covariance matrix and means."""

        def check_diag(diag):
            if not np.allclose(diag, 2.0 / settings.HBAR):
                raise ValueError("The covariance matrix is not for a Gaussian pure state.")

        if cov.shape == 2:
            symplectic, diag = williamson(cov)
            check_diag(diag)
            symplectic = math.expand_dims(symplectic, axis=0)
        elif cov.shape == 3:
            symplectic = cov
            for i in range(cov.shape[0]):
                symplectic[i, :], diag = williamson(cov[i, :])
                check_diag(diag)
        return cls(symplectic, means)
