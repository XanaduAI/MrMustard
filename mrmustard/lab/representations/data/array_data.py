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
        super().__init__()


    #@njit
    def __neg__(self) -> Data:
        r"""
        Returns the negative of the object

        Args:
            NA

        Returns:
            The negative object
        """
        return self.__class__(array= -self.array)
        


    def __eq__(self, other:ArrayData) -> bool:
        r"""
        Compares two ArrayData objects

        Args:
            other (ArrayData) : the object being compared

        Returns:
            True if both objects are equal, False otherwise
        """

        try:
            return super().same(X=[self.array], Y=[other.array])
        
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e



    #@njit
    def __add__(self, other:ArrayData) -> ArrayData:
        r"""
        Adds two ArrayData objects

        Args:
            other (ArrayData): the object to be added

        Returns:
            An array resulting form adding the two objects
        """

        try:
            return self.__class__(array=self.array + other.array)
        
        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e
            


    #@njit
    def __sub__(self, other:ArrayData) -> ArrayData:
        r"""
        Subtracts two Data objects

        Args:
            other (ArrayData): the object to be subtracted

        Returns:
            An array resulting form subtracting two objects
        """
        self.__add__(-other)


    def __truediv__(self, other:Union[Scalar, ArrayData]) -> ArrayData:
        r"""
        Divides two Data objects

        Args:
            other (Union[Scalar, ArrayData]): the object to be divided by

        Returns:
            An array resulting form dividing two objects
        """
        raise NotImplementedError()


    #@njit(parallel=True)
    def __mul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        r"""
        Multiplies two ArrayData objects or an ArrayData and a Scalar 

        Args:
            other (Union[Scalar, ArrayData]): the object to be multiplied with

        Returns:
            An object of the common child Data class resulting form multiplying two objects
        """

        try:
            return self.__class__(array=self.array * other.array)
        
        except AttributeError:

            try: # if it's not an array, we try a Number
                return self.__class__(array=self.array * other)
            
            except TypeError as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
            

    
    #@njit(parallel=True)
    def __rmul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        r""" See __mul__, object have commutative multiplication."""
        return self.__mul__(other=other)



    #@njit
    def __truediv__(self, other:Union[Scalar, ArrayData]) -> ArrayData:
        r"""
        Divides two Data objects

        Args:
            other (Union[Scalar, ArrayData]): the object to be divided with

        Returns:
            An array resulting form dividing two objects or the object by a Scalar
        """
        raise NotImplementedError



    #@njit(parallel=True)
    def __and__(self, other:ArrayData) -> ArrayData: # TODO : check this, it's an outer product, how can it return an Array?
        r"""
        Performs the tensor product between two Data objects

        Args:
            other (Data): the object to be tensor-producted with

        Returns:
            A matrix resulting form tensoring two objects
        """

        try:
            return self.__class__(array=np.outer(self.array, other.array))
        
        except AttributeError as e:
         raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e             



    #@njit(parallel=True)
    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> ArrayData:
        r"""
        Performs the simplification of the object, using some data compression

        Args:
            rtol (float): the relative tolerance for numpy's `allclose`
            atol (float): the absolute tolerance for numpy's `allclose`

        Returns:
            A simplified object
        """
        raise NotImplementedError() # TODO: implement
    