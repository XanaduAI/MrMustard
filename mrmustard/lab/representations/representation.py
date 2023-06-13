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

from abc import ABC, abstractmethod, abstractproperty
from mrmustard.typing import RealMatrix, RealVector, Scalar

class Representation(ABC):
    r""" Abstract parent class for the different Representation of quantum states. """
    
    @abstractproperty
    def purity(self) -> Scalar:
        r""" The purity of the state. """
        raise NotImplementedError()
    

    @abstractproperty
    def norm(self) -> float:
        r""" The norm of the state. """
        raise NotImplementedError()
    

    @abstractproperty
    def von_neumann_entropy(self) -> float:
        r""" The Von Neumann entropy of the state. 
        
        For a pure state, we expect the Von Neumann entropy to be 0.

        Reference: (https://arxiv.org/pdf/1110.3234.pdf), Equations 46-47.        
        """
        raise NotImplementedError()
    

    @abstractproperty
    def number_means(self) -> RealVector:
        r""" The photon number means vector. """
        raise NotImplementedError()


    @abstractproperty
    def number_cov(self) -> RealMatrix:
        r""" The photon number covariance matrix"""
        raise NotImplementedError()


    @abstractmethod
    def number_stdev(self) -> int: # TODO : add doc
        raise NotImplementedError()


    @abstractmethod
    def probability(self) -> float: # TODO : add doc
        raise NotImplementedError()


    @abstractmethod
    def _repr_markdown_(self) -> str:
        raise NotImplementedError()