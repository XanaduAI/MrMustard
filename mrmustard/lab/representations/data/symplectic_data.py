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
from mrmustard.lab.representations.data import MatVecData
from mrmustard.typing import Scalar

class SymplecticData(MatVecData):
    """ Symplectic matrix-like data for certain Representation objects.

    Args:
        mat:
        vec:
        coeffs:
    """

    def __init__(self, mat, vec, coeffs) -> None:
        super().__init__(mat=mat, vec=vec, coeffs=coeffs)


    @property
    def mean(self) -> np.array:
        return self.vec


    # TODO : implement
    def __truediv__(self, other:Union[Scalar, SymplecticData]) -> SymplecticData:
        raise NotImplementedError()


    def __mul__(self, other:Union[Scalar, SymplecticData]) -> SymplecticData:
        if isinstance(other, Scalar):
            return self.__class__(mat=self.mat, vec=self.vec, coeffs=self.coeffs)
        
        else: # TODO : use MM's math module where possible
            raise NotImplementedError() # TODO : implement (is the below correct?)
            # try:
            #     return self.__class__(mat=np.matmul(self.mat, other.mat), 
            #                         mean=np.multiply(self.mean, other.mean),
            #                         coeff=np.multiply(self.coeff, other.coeff))
            
            # except AttributeError as e:
            # raise TypeError(f"Cannot tensor {self.__class__} and {other.__class__}.") from e
