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
from mrmustard.typing import Scalar

class Data(ABC):
    r"""
    Abstract class, parent to the different types of data that can encode a quantum State 
    and Representation.
    """

    def __init__(self, nb_modes) -> None:
        self.nb_modes = nb_modes
        super().__init__()

        
    def same(self, X: List[Union[List[Scalar],Scalar]], 
             Y: List[Union[List[Scalar],Scalar]], 
             rtol:float=1e-6, atol:float=1e-6) -> bool:
        r"""
        Method to compare two lists of elements

        Args:
            X (List[Union[List[Scalar],Scalar]])    : list of elements to be compared with ys
            Y (List[Union[List[Scalar],Scalar]])    : list of elements to be compared with xs
            rtol (float)                            : relative tolerance parameter for Numpy's 
                                                      `all_close`
            atol (float)                            : absolute tolerance parameter for Numpy's 
                                                      `all_close`

        Returns:
            True if all the elements compared are same within rtol/atol bounds, False otherwise
        """
        return all([np.allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(X, Y)])

    
    @abstractmethod
    def __eq__(self, other:Data) -> bool:
        r"""
        Abstract method comparing two Data objects

        Args:
            other (Data)    : the object being compared

        Returns:
            True if both objects are equal, False otherwise
        """
        raise NotImplementedError()

    
     @abstractmethod
    def __neg__(self) -> Data:
        r""" Negative of the object """
        raise NotImplementedError()


    @abstractmethod
    def __add__(self, other: Data, rtol:float=1e-6, atol:float=1e-6) -> Data:
        r"""
        Abstract method adding two Data objects

        Args:
            other (Data): the object to be added
            rtol (float): relative tolerance parameter for Numpy's `all_close`
            atol (float): absolute tolerance parameter for Numpy's `all_close`

        Returns:
            An object of the common child Data class resulting form adding two objects
        """
        raise NotImplementedError()



    @abstractmethod
    def __sub__(self, other: Data, rtol:float=1e-6, atol:float=1e-6) -> Data:
        r"""
        Abstract method subtracting two Data objects

        Args:
            other (Data): the object to be subtracted
            rtol (float): relative tolerance parameter for Numpy's `all_close`
            atol (float): absolute tolerance parameter for Numpy's `all_close`

        Returns:
            An object of the common child Data class resulting form subtracting two objects
        """
        raise NotImplementedError()



    @abstractmethod
    def __truediv__(self, other:Union[Scalar, Data]) -> Data:
        r"""
        Abstract method dividing two Data objects

        Args:
            other (Data): the object to be divided with
            rtol (float): relative tolerance parameter for Numpy's `all_close`
            atol (float): absolute tolerance parameter for Numpy's `all_close`

        Returns:
            An object of the common child Data class resulting form dividing two objects
        """
        raise NotImplementedError()



    @abstractmethod
    def __mul__(self, other:Union[Scalar, Data]) -> Data:
        r"""
        Abstract method multiplying two Data objects

        Args:
            other (Data): the object to be multiplied with
            rtol (float): relative tolerance parameter for Numpy's `all_close`
            atol (float): absolute tolerance parameter for Numpy's `all_close`

        Returns:
            An object of the common child Data class resulting form multiplying two objects
        """
        raise NotImplementedError()



    @abstractmethod
    def __and__(self, other:Data) -> Data:
        r"""
        Abstract method performing the tensor product between two Data objects

        Args:
            other (Data): the object to be tensor-producted with
            rtol (float): relative tolerance parameter for Numpy's `all_close`
            atol (float): absolute tolerance parameter for Numpy's `all_close`

        Returns:
            An object of the common child Data class resulting form tensoring two objects
        """
        raise NotImplementedError()



    @abstractmethod
    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> Data:
        r"""Optimises the representation of an object by performing some custom data compression"""
        raise NotImplementedError()


