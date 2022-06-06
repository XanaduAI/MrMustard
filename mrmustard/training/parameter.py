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

"""
The classes in this module represent parameters passed to the
quantum operations represented by :class:`~.Parametrized` subclasses.

Parameter types
---------------
There are three basic types of parameters:
1. **Numerical parameters** (bound and fixed): An immediate, immutable numerical python objects
   (float, complex, int, numerical array). Implemented as-is, not encapsulated in a class.
   They are dynamically assigned to properties of the class.
2. **Trainable parameters** (bound but not fixed): These are parameters that are updated by
   the optimization procedure. There are three types of trainable parameters which define the
   optimization procedure: symplectic, euclidian and orthogonal.
3. **Constant parameters** (bound and fixed): This class of parameters belong to the autograd
   backend but remain fixed during the optimization procedure.
"""
# pylint: disable=super-init-not-called

from abc import ABC, abstractmethod

from mrmustard.math import Math
from mrmustard.types import Tensor, Tuple

math = Math()


class Parameter(ABC):
    @abstractmethod
    def __init__(self, value, name, owner=None) -> None:
        pass

    @property
    def value(self) -> Tensor:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def type(self) -> str:
        return self.__class__.__name__.lower()


class Trainable(Parameter, ABC):
    @abstractmethod
    def __init__(self, value, name, owner=None) -> None:
        pass


class Symplectic(Trainable):
    def __init__(self, value, name, owner=None) -> None:
        self._value = (
            value
            if math.from_backend(value) and math.is_trainable(value)
            else math.new_variable(value, None, name)
        )
        self._name = name
        self._owner = owner


class Euclidean(Trainable):
    def __init__(self, value, bounds, name, owner=None) -> None:
        self._value = (
            value
            if math.from_backend(value) and math.is_trainable(value)
            else math.new_variable(value, bounds, name)
        )
        self._name = name
        self._owner = owner
        self.bounds = bounds


class Orthogonal(Trainable):
    def __init__(self, value, name, owner=None) -> None:
        self._value = (
            value
            if math.from_backend(value) and math.is_trainable(value)
            else math.new_variable(value, None, name)
        )
        self._name = name
        self._owner = owner


class Constant(Parameter):
    def __init__(self, value, name, owner=None) -> None:
        self._value = (
            value
            if math.from_backend(value) and not math.is_trainable(value)
            else math.new_constant(value, name)
        )
        self._name = name
        self._owner = owner


def create_parameter(value, name, is_trainable=False, bounds=None, owner=None) -> Trainable:

    if not is_trainable:
        return Constant(value, name, owner)

    if name.startswith("symplectic"):
        return Symplectic(value, name, owner)

    if name.startswith("orthogonal"):
        return Orthogonal(value, name, owner)

    return Euclidean(value, bounds, name, owner)
