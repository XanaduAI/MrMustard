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

class Representation(ABC):

    def __init__(self):
        super().__init__()

    
    @abstractmethod
    def number_means():
        pass


    @abstractmethod
    def number_cov():
        pass


    @abstractmethod
    @property
    def purity():
        pass


    @abstractmethod
    def number_stdev():
        pass


    @abstractmethod
    @property
    def norm():
        pass


    @abstractmethod
    def probability():
        pass


    @abstractmethod
    @property
    def von_neumann_entropy():
        pass


    @abstractmethod
    def __add__():
        pass
        

    @abstractmethod
    def __eq__():
        pass


    @abstractmethod
    def __rmul__():
        pass


    @abstractmethod
    def __truediv__():
        pass


    @abstractmethod
    def _repr_markdown_():
        pass

     
