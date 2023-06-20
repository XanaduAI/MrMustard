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
from typing import Union
from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.typing import RealMatrix, Scalar, RealVector
from typing import Optional
from thewalrus.symplectic import is_symplectic

class SymplecticData(MatVecData):
    """ Symplectic matrix-like data for certain Representation objects.

    Args:
        symplectic (Matrix): symplectic matrix with qqpp-ordering
        displacement (Vector): the real displacement vector :math:`\bm{d} = \sqrt{2\hbar}[\Re(\alpha), \Im(\alpha)]`
        coeffs (Scalar) : default to be 1.
    """

    def __init__(self, symplectic: RealMatrix, displacements: RealVector, coeffs: Scalar = 1.0) -> None:
        #Check if it is a symplectic matrix
        if is_symplectic(symplectic.numpy()):
            super().__init__(mat=symplectic, vec=displacement, coeffs=coeffs)
        else:
            raise ValueError("The matrix given is not symplectic.")


    @property
    def symplectic(self) -> np.array:
        return self.mat
    

    @property
    def displacements(self) -> np.array:
        return self.vec



    def __mul__(self, other:Scalar) -> SymplecticData:
        if isinstance(other, SymplecticData):
            raise TypeError("Symplectic can only be multiplied by a scalar")
        else:
            new_coeffs = self.coeffs * other
            return self.__class__(symplectic= self.symplectic, displacements= self.displacements, 
                                  coeffs= new_coeffs)