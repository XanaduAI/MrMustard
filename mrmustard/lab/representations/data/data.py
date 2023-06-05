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

from abc import ABC
from typing import List

class Data(ABC):

    def __init__(self) -> None:
        super().__init__()

    
    
    @abstractmethod
    def __neg__(self): #implem here!
        raise NotImplementedError()

    
    
    @abstractmethod
    def __eq__(self, other: List[Data], rtol=1e-6, atol=1e-6):
        raise NotImplementedError()



    @abstractmethod
    def __add__(self, other: Data, rtol=1e-6, atol=1e-6):
        raise NotImplementedError()



    @abstractmethod
    def __sub__(self, other: Data, rtol=1e-6, atol=1e-6):
        raise NotImplementedError()



    @abstractmethod
    def __truediv__(self):
        raise NotImplementedError()



    @abstractmethod
    def __mul__(self, other:Union[Number, Data]):
        raise NotImplementedError()



    @abstractmethod
    def __and__(self, other:Data):
        raise NotImplementedError()



    @abstractmethod
    def simplify(self, rtol=1e-6, atol=1e-6):
        raise NotImplementedError()


