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
from mrmustard.typing import R, Scalar

class SymplecticData(MatVecData):
    """ Symplectic matrix-like data for certain Representation objects.

    Args:
        mat:
        vec:
        coeffs:
    """

    def __init__(self, symplectic, displacement:R, coeffs) -> None:
        # TODO : if no coeff given, they are all 1, and change args! coeff is optional
        super().__init__(mat=symplectic, vec=displacement, coeffs=coeffs)
        # TODO : ensure mat is symplectic! 


    @property
    def symplectic(self) -> np.array:
        return self.mat
    
    @property
    def displacement(self) -> np.array:
        return self.vec


    def __truediv__(self, other:Scalar) -> SymplecticData:
        return self.__class__(symplectic=self.mat, displacement=self.vec, coeffs=self.coeffs/other)


    def __mul__(self, other:Scalar) -> SymplecticData:
        return self.__class__(symplectic=self.mat, displacement=self.vec, coeffs=self.coeffs*other)