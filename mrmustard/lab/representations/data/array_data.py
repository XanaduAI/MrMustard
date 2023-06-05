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
from typing import Union
from mrmustard.representations.data import Data
from mrmustard.typing import Scalar

__all__ = [ArrayData]

class ArrayData(Data):

    def __init__(self, array):

        self.array = array
        self.cutoffs = array.shape
        super().__init__()


    @njit
    def __neg__(self) -> ArrayData:
        return self.__class__(array= -self.array) # Note : the almost invisible "-" sign
        


    def __eq__(self, other: ArrayData, rtol:float=1e-6, atol:float=1e-6) -> bool:

        try:
            return np.allclose(self.array, other.array, rtol=rtol, atol=atol)
        
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e



    @njit
    def __add__(self, other:ArrayData) -> ArrayData:

        try:
            return self.__class__(array=self.array + other.array)
        
        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e
            


    @njit
    def __sub__(self, other: ArrayData) -> ArrayData:
        self.__add__(-other)



    @njit(parallel=True)
    def __mul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:

        try:
            return self.__class__(array=self.array * other.array)
        
        except AttributeError:
            try: # if it's not an array, we try a Number
                return self.__class__(array=self.array * other)
            
            except TypeError as e:
                raise TypeError(f"Cannot multiply/divide {self.__class__} and {other.__class__}."
                                ) from e



    @njit
    def __truediv__(self, Union[Scalar, ArrayData]) -> ArrayData:
        self.__mul__(1/other)



    @njit(parallel=True)
    def __and__(self, other:ArrayData) -> ArrayData:

        try:
            return self.__class__(array=np.outer(self.array, other.array))
        
        except AttributeError as e:
         raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e             



    @njit(parallel=True)
    def simplify(self) -> ArrayData: # TODO: implement
        raise NotImplementedError() 
    