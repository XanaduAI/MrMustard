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
#from numba import njit
import numpy as np
from typing import Union
from mrmustard.lab.representations.data import Data
from mrmustard.math import Math
from mrmustard.typing import Scalar

math = Math()

class ArrayData(Data):

    def __init__(self, array) -> None:

        self.array = array
        self.cutoffs = array.shape

        super().__init__(nb_modes=None) # TODO : fix this with an actual value


    #@njit
    def __neg__(self) -> Data:
        return self.__class__(array= -self.array)
        


    def __eq__(self, other: ArrayData) -> bool:

        try:
            return super().same(X=[self.array], Y=[other.array])
        
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e



    #@njit
    def __add__(self, other:ArrayData) -> ArrayData:

        try:
            return self.__class__(array=self.array + other.array)
        
        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e
            


    #@njit
    def __sub__(self, other: ArrayData) -> ArrayData:
        self.__add__(-other)



    #@njit(parallel=True)
    def __mul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:

        try:
            return self.__class__(array=self.array * other.array)
        
        except AttributeError:
            try: # if it's not an array, we try a Number
                return self.__class__(array=self.array * other)
            
            except TypeError as e:
                raise TypeError(f"Cannot multiply/divide {self.__class__} and {other.__class__}."
                                ) from e
            

    
    #@njit(parallel=True)
    def __rmul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        return self.__mul__(other=other)



    #@njit
    def __truediv__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        self.__mul__(other = 1/other)



    #@njit(parallel=True)
    def __and__(self, other:ArrayData) -> ArrayData:

        try:
            return self.__class__(array=np.outer(self.array, other.array))
        
        except AttributeError as e:
         raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e             



    #@njit(parallel=True)
    def simplify(self) -> ArrayData: # TODO: implement
        raise NotImplementedError() 
    