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
from typing import Union, List
from mrmustard.lab.representations.data import ArrayData
from mrmustard.math import Math
from mrmustard.typing import Scalar

math = Math()

class WavefunctionArrayData(ArrayData):

    def __init__(self, qs, array) -> None:
        r"""
        Initializes the wavefunction with the q-variable points qs and the corresponding values in array

        Args:
            qs (Array): q-variable points 
            array (Array): q-Wavefunction values correspoidng qs
        
        Returns:
            None
        """
        super().__init__(array=array)
        self.qs = qs

    
    @property
    def cutoffs(self) -> Union[int, List[int]]:
        r"""
        Returns the cutoffs of the q-Wavefunction
        """
        return self.array.shape


    #@njit
    def __neg__(self) -> WavefunctionArrayData:
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
            return super().same(X=[self.array], Y=[other.array]) and super.same(X=[self.qs], Y=[other.qs])
        
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e



    #@njit
    def __add__(self, other:ArrayData) -> WavefunctionArrayData:
        r"""
        Adds two WavefunctionArrayData objects' array

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
    def __sub__(self, other:ArrayData) -> WavefunctionArrayData:
        r"""
        Subtracts two Data objects

        Args:
            other (ArrayData): the object to be subtracted

        Returns:
            An array resulting form subtracting two objects
        """
        self.__add__(-other)


    #@njit
    def __truediv__(self, other:Union[Scalar, ArrayData]) -> WavefunctionArrayData:
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



    #@njit(parallel=True)
    def __and__(self, other:WavefunctionArrayData) -> WavefunctionArrayData: # TODO : check this, it's an outer product, how can it return an Array?
        r"""
        Performs the tensor product between two Data objects

        Args:
            other (Data): the object to be tensor-producted with

        Returns:
            A matrix resulting form tensoring two objects
        """

        if self.qs == other.qs:
            return self.__class__(array=np.outer(self.array, other.array))
        else:
            raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__} because the q-variable points are not the same.") from e             



    #@njit(parallel=True)
    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> WavefunctionArrayData:
        r"""
        Performs the simplification of the object, using some data compression

        Args:
            rtol (float): the relative tolerance for numpy's `allclose`
            atol (float): the absolute tolerance for numpy's `allclose`

        Returns:
            A simplified object
        """
        raise NotImplementedError() # TODO: implement
    