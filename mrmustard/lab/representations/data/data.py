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
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union
from mrmustard.typing import Scalar, Vector

class Data(ABC):
    r""" Abstract parent class to the different types of data that can encode a quantum State and
      Representation.
    """

    def __init__(self) -> None:
        pass


    @staticmethod   
    def same(
             X: List[Union[List[Vector],List[Scalar]]],
             Y: List[Union[List[Vector],List[Scalar]]]
             ) -> bool:
        r""" Method to compare two sets of elements

        Args:
            X: list of elements to be compared with ys
            Y: list of elements to be compared with xs

        Returns:
            True if all the elements compared are same within numpy's default rtol/atol, False 
            otherwise.
        """
        return all([np.allclose(x, y) for x, y in zip(X, Y)])


    @abstractmethod
    def __neg__(self) -> Data:
        raise NotImplementedError()
    

    @abstractmethod
    def __eq__(self, other:Data) -> bool:
        raise NotImplementedError()


    @abstractmethod
    def __add__(self, other: Data) -> Data:
        raise NotImplementedError()


    @abstractmethod
    def __sub__(self, other: Data) -> Data:
        raise NotImplementedError()


    @abstractmethod
    def __truediv__(self, other:Union[Scalar, Data]) -> Data:
        raise NotImplementedError()
    

    @abstractmethod
    def __mul__(self, other:Union[Scalar, Data]) -> Data:
        raise NotImplementedError()
    

    def __rmul__(self, other:Union[Scalar, Data]) -> Data:
        return self.__mul__(other=other)


    @abstractmethod
    def __and__(self, other:Data) -> Data:
        r""" Performs the tensor product between the two objects. """
        raise NotImplementedError()


    @abstractmethod
    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> Data:
        r""" Simplifies the object by performing some data compression.

        Args:
            rtol: the relative tolerance for numpy's `allclose`
            atol: the absolute tolerance for numpy's `allclose`

        Returns:
            An object of the same input class, simplified
        """
        raise NotImplementedError()


