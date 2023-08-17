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

from mrmustard.lab.representations.wigner import Wigner
from mrmustard.lab.representations.data.symplectic_data import SymplecticData
from mrmustard.typing import Matrix, Vector, Scalar
from mrmustard.math import Math

math = Math()
from mrmustard import settings
from thewalrus.symplectic import is_symplectic


class WignerKet(Wigner):
    r"""Wigner representation of a ket state.

    Args:
        symplectic: symplectic matrix
        displacement: dispalcement vector
        coeffs: coefficients (complex)
    """

    def __init__(self, symplectic: Matrix, displacement: Vector, coeffs: Scalar = 1.0) -> None:
        # Check the symplecticity of the matrix
        if not is_symplectic(symplectic):
            raise ValueError("The matrix is not symplectic!")
        self.data = SymplecticData(symplectic=symplectic, displacement=displacement, coeffs=coeffs)

    @property
    def cov(self):
        return (
            settings.HBAR
            / 2
            * math.matmul(self.data.symplectic, math.transpose(self.data.symplectic))
        )

    @property
    def means(self):
        return self.data.displacement

    @property
    def purity(self) -> float:
        return 1.0
