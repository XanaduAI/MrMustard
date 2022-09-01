# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the abstract Representation class."""



from abc import ABC, abstractmethod

class Representation(ABC):

    @abstractmethod # so that Representation can't be instantiated but subclasses can still call super().__init__()
    def __init__(self, representation: Representation):
        if isinstance(representation, Characteristic):
            try:
                self.from_characteristic_gaussian(representation)
            except AttributeError:
                self.from_characteristic_array(representation)
        elif isinstance(representation, Wigner):
            try:
                self.from_wigner_gaussian(representation)
            except AttributeError:
                self.from_wigner_array(representation)
        elif isinstance(representation, Bargmann):
            try:
                self.from_bargmann_gaussian(representation)
            except AttributeError:
                self.from_bargmann_array(representation)
        elif isinstance(representation, Fock):
            try:
                self.from_fock_gaussian(representation)
            except AttributeError:
                self.from_fock_array(representation)
        elif isinstance(representation, Position):
            try:
                self.from_position_gaussian(representation)
            except AttributeError:
                self.from_position_array(representation)
        elif isinstance(representation, Momentum):
            try:
                self.from_momentum_gaussian(representation)
            except AttributeError:
                self.from_momentum_array(representation)
        else:
            raise ValueError("Cannot convert representation {representation.__class__.__name__} to {self.__class__.__name__}")

    @abstractmethod
    def from_characteristic_gaussian(self, characteristic: Characteristic):
        raise NotImplementedError(f'Converting from characteristic gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_characteristic_array(self, characteristic: Characteristic):
        raise NotImplementedError(f'Converting from characteristic array to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_wigner_gaussian(self, wigner: Wigner):
        raise NotImplementedError(f'Converting from wigner gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_wigner_array(self, wigner: Wigner):
        raise NotImplementedError(f'Converting from wigner array to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_bargmann_gaussian(self, bargmann: Bargmann):
        raise NotImplementedError(f'Converting from bargmann gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_bargmann_array(self, bargmann: Bargmann):
        raise NotImplementedError(f'Converting from bargmann array to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_fock_gaussian(self, fock: Fock):
        raise NotImplementedError(f'Converting from fock gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_fock_array(self, fock: Fock):
        raise NotImplementedError(f'Converting from fock array to {self.__class__.__name__} is not implemented')
    
    @abstractmethod
    def from_position_gaussian(self, position: Position):
        raise NotImplementedError(f'Converting from position gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_position_array(self, position: Position):
        raise NotImplementedError(f'Converting from position array to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_momentum_gaussian(self, momentum: Momentum):
        raise NotImplementedError(f'Converting from momentum gaussian to {self.__class__.__name__} is not implemented')

    @abstractmethod
    def from_momentum_array(self, momentum: Momentum):
        raise NotImplementedError(f'Converting from momentum array to {self.__class__.__name__} is not implemented')
        