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

import numpy as np
from numba import njit
from mrmustard.representations.data import MatVecData

class SymplecticData(MatVecData):

    def __init__(self, mat, mean, coeff) -> SymplecticData:
        super().__init__(mat=mat, vec=mean, coeff=coeff)



        @property
        def mean(self) -> np.array:
            return self.vec


        
        def __truediv__(self): # TODO : implement
            raise NotImplementedError()



        @njit
        def __mul__(self, other:Union[Number, Data]) -> SymplecticData:

            if self.__class__ != other.__class__ and type(other) != Number:
                raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.")
            
            else:
            
                try:
                    return SymplecticData(mat=np.matmul(self.mat, other.mat), 
                                        mean=np.multiply(self.mean, other.mean),
                                        coeff=np.multiply(self.coeff, other.coeff))
                
                except AttributeError:
                    return SymplecticData(mat=self.mat * other, 
                                          mean=self.mean*other, 
                                          coeff=self.coeff*other)
