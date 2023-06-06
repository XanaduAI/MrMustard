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
from typing import List, Union
from mrmustard.typing import Scalar

class Data(ABC):

    def __init__(self) -> None:
        super().__init__()

        
    def same(self, X: List[Scalar], Y: List[Scalar], rtol:float=1e-6, atol:float=1e-6) -> bool:
        return all([np.allclose(x, y) for x, y in zip(X, Y)])

    
    @abstractmethod
    def __eq__(self) -> bool:
        raise NotImplementedError()

    
     @abstractmethod
    def __neg__(self) -> Data:
        raise NotImplementedError()


    @abstractmethod
    def __add__(self, other: Data, rtol:float=1e-6, atol:float=1e-6) -> Data:
        raise NotImplementedError()



    @abstractmethod
    def __sub__(self, other: Data, rtol:float=1e-6, atol:float=1e-6) -> Data:
        raise NotImplementedError()



    @abstractmethod
    def __truediv__(self, other:Union[Scalar, Data]) -> Data:
        raise NotImplementedError()



    @abstractmethod
    def __mul__(self, other:Union[Scalar, Data]) -> Data:
        raise NotImplementedError()



    @abstractmethod
    def __and__(self, other:Data) -> Data:
        raise NotImplementedError()



    @abstractmethod
    def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> Data:
        raise NotImplementedError()


