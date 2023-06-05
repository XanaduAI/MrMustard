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
from mrmustard.representations.data import Data

class ArrayData(Data):

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.cutoffs = array.shape



    @njit(parallel=True)
    def __eq__(self, other: ArrayData, rtol=1e-6, atol=1e-6) -> bool:

        if self.__class__ != other.__class__:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.")
        
        else:
            return np.allclose(self.array, other.array, rtol=rtol, atol=atol)



    
    def __add__(self, other:ArrayData) -> ArrayData:

        if self.__class__ != other.__class__:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.")
        
        else:
            return ArrayData(array=self.array + other.array)



    def __sub__(self, other: ArrayData) -> ArrayData:

        if self.__class__ != other.__class__:
            raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.")
        
        else:
            return ArrayData(array=self.array - other.array)



    
    def __truediv__(self, other: ArrayData) -> ArrayData:

        if self.__class__ != other.__class__:
            raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.")
        
        else:
            return ArrayData(array=np.true_divide(self.array, other.array))



    @njit(parallel=Tru)
    def __mul__(self, other: Union[Number, ArrayData]) -> ArrayData:

        if self.__class__ != other.__class__ and type(other) != Number:
            raise TypeError(f"Cannot perform multiplication here.")
        
        else:
            try:
                return ArrayData(array=self.array * other.array)

            except AttributeError: # it is a Number
                return ArrayData(array=self.array * other)



    def __neg__(self): #implem here
        return ArrayData(array=-self.array)



    @njit(parallel=True)
    def __and__(self, other:ArrayData) -> ArrayData:
         
         if self.__class__ != other.__class__:
            raise TypeError(f"Cannot do tensor product on {self.__class__} and {other.__class__}.")
         
         else:
             return ArrayData(array=np.outer(self.array, other.array))
             



    @abstractmethod
    def simplify(self): # TODO: implement
        raise NotImplementedError() 
    