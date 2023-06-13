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
from typing import Union, List
from mrmustard.lab.representations.data import ArrayData
from mrmustard.math import Math
from mrmustard.typing import Scalar

math = Math()

class WavefunctionArrayData(ArrayData):
    r""" Encapsulates the q-variable points and correspodning values.

    qs: q-variable points 
    array: q-Wavefunction values correspoidng qs
    """

    def __init__(self, qs:np.array, array:np.array) -> None:
        super().__init__(array=array)
        self.qs = qs

    
    @property
    def cutoffs(self) -> Union[int, List[int]]:
        r""" Cutoffs of the q-Wavefunction. """
        return self.array.shape
    

    def _qs_is_same(self, other:WavefunctionArrayData) -> bool:
        r""" Compares the qs of two WavefunctionArrayData objects. """
        try:
            return True if np.allclose(self.qs, other.qs) else False
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e


    def __neg__(self) -> WavefunctionArrayData:
        return self.__class__(array= -self.array, qs=self.qs) # TODO
        

    def __eq__(self, other:ArrayData) -> bool:
        try:
            return (super().same(X=[self.array], Y=[other.array]) 
                    and super.same(X=[self.qs], Y=[other.qs]))
        
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e


    def __add__(self, other:ArrayData) -> WavefunctionArrayData:
        if self._qs_is_same(other):
            try:
                return self.__class__(array=self.array + other.array, qs=self.qs) # TODO

            except AttributeError as e:
                raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}."
                                ) from e
        else:
            raise ValueError ("The two wave functions must have the same qs. ")
            

    def __sub__(self, other:ArrayData) -> WavefunctionArrayData:
        self.__add__(-other)


    def __truediv__(self, other:Union[Scalar, ArrayData]) -> WavefunctionArrayData:
        raise NotImplementedError()


    def __mul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        if self._qs_is_same(other):
            try:
                return self.__class__(array=self.array * other.array, qs=self.qs)
            
            except AttributeError:
                try: # if it's not an array, we try a Number
                    return self.__class__(array=self.array * other, qs=self.qs)
                except TypeError as e:
                    raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}."
                                    ) from e
        else:
            raise ValueError ("The two wave functions must have the same qs. ")
            

    def __rmul__(self, other: Union[Scalar, ArrayData]) -> ArrayData:
        return self.__mul__(other=other)


    def __and__(self, other:WavefunctionArrayData) -> WavefunctionArrayData:
        if self._qs_is_same(other):
            return self.__class__(array=np.outer(self.array, other.array), qs=self.qs)
        else:
            raise ValueError ("The two wave functions must have the same qs. ")


    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> WavefunctionArrayData:
        raise NotImplementedError() # TODO: implement
    