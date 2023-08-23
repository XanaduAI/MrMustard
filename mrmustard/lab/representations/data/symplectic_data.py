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

from itertools import product
from typing import Optional

import numpy as np
from thewalrus.quantum.gaussian_checks import is_symplectic

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, Matrix, RealVector, Scalar

math = Math()


class SymplecticData(MatVecData):
    """Symplectic matrix-like data for certain Representation objects.

    Here the displacement vector is defined as:
    :math:`\bm{d} = \sqrt{2\hbar}[\Re(\alpha), \Im(\alpha)]`

    Args:
        symplectic (Batch[Matrix]):         symplectic matrix with qqpp-ordering
        displacement (Batch[RealVector]):   the real displacement vector
        coeffs (Optional[Batch[Scalar]]):   default to be 1.
    """

    def __init__(
        self,
        symplectic: Batch[Matrix],
        displacement: Batch[RealVector],
        coeffs: Optional[Batch[Scalar]] = None,
    ) -> None:
        super().__init__(mat=symplectic, vec=displacement, coeffs=coeffs)
        for mat in self.mat:
            if is_symplectic(math.asnumpy(mat)) == False:
                raise ValueError("The matrix given is not symplectic.")

        # reaching here means no matrix is non-symplectic

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
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: SymplecticData) -> SymplecticData:
        "symplectic tensor product as block-wise block diag concatenation"
        if isinstance(other, SymplecticData):
            new_symplectics = [
                math.symplectic_tensor_product(s1, s2)
                for s1, s2 in product(self.symplectic, other.symplectic)
            ]
            new_displacements = []
            d = len(self.displacement[0]) // 2
            for d1, d2 in product(self.displacement, other.displacement):
                new_displacements.append(np.concatenate([d1[:d], d2[:d], d1[d:], d2[d:]]))
            new_displacements = math.astensor(new_displacements)
            new_coeffs = math.reshape(math.outer(self.coeffs, other.coeffs), -1)
            return self.__class__(new_symplectics, new_displacements, new_coeffs)
        else:
            raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.")
