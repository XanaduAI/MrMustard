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

from typing import Optional

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, Matrix, Scalar, RealVector
from thewalrus.quantum.gaussian_checks import is_symplectic

math = Math()


class SymplecticData(MatVecData):
    """Symplectic matrix-like data for certain Representation objects.

    Here the displacement vector is defined as:
    :math:`\bm{d} = \sqrt{2\hbar}[\Re(\alpha), \Im(\alpha)]`

    Args:
        symplectic:     symplectic matrix with qqpp-ordering
        displacement:   the real displacement vector
        coeffs:         default to be 1.
    """

    def __init__(self, symplectic: Batch[Matrix],
                 displacement: Batch[RealVector], 
                 coeffs: Optional[Batch[Scalar]]=None
                 ) -> None:
        for mat in symplectic:
            if is_symplectic(math.asnumpy(mat)) == False:
                raise ValueError("The matrix given is not symplectic.")

        # reaching here means no matrix is non-symplectic
        super().__init__(mat=symplectic, vec=displacement, coeffs=coeffs)            

    @property
    def symplectic(self) -> np.array:
        return self.mat

    @property
    def displacement(self) -> np.array:
        return self.vec

    def __mul__(self, other: Scalar) -> SymplecticData:
        if isinstance(other, SymplecticData):
            raise TypeError("Symplectic can only be multiplied by a scalar")
        else:
            try:  # Maybe other is a scalar
                new_coeffs = self.coeffs * other
                return self.__class__(self.symplectic, self.displacement, new_coeffs)
            except TypeError as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
