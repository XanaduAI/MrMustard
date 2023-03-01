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
The classes in this module contain the :class:`Parameter` passed to the
quantum operations represented by :class:`Parametrized` subclasses.

Parameter types
---------------
There are three basic types of parameters:

1. **Numerical parameters** (bound and fixed): An immediate python object
   (float, complex, int, list, numerical array, ...). Implemented as-is, not encapsulated in a class
   and has the typical python behaviour. They are assigned to properties of the relevant class.
   For example,

   .. code-block::

        class Gate():
            def __init__(self, modes: List):
                self._modes = modes

2. **Trainable parameters** (bound but not fixed): These are parameters that are updated by
   the optimization procedure. Tipically, this are defined via arguments of the
   :class:`Parametrized` class constructor.

      .. code-block::

        class Gate(Parametrized):
            def __init__(self, r: float, modes: List, r_trainable: bool):
                super.__init__(r=r, r_trainable=r_trainable)
                self._modes = modes

        gate = Gate(r=0, modes=[1], r_trainable=True)
        gate._r     # access the dynamically assigned property of the trainable parameter
        isinstance(gate._r, Parameter)      # evaluates to True

    The dynamically assigned property is an instance of :class:`Parameter` and contains the
    ``value`` property which is a tensor of the autograd backend.

    There are four types of trainable parameters: symplectic, euclidean, unitary and orthogonal.
    Each type defines a different optimization procedure on the :py:training: module.

    .. code-block::

        class SymplecticGate(Parametrized):
            def __init__(self, symplectic: Tensor):
                super.__init__(symplectic=symplectic, symplectic_trainable=True)

        class EuclideanGate(Parametrized):
            def __init__(self, euclidean: Tensor):
                super.__init__(euclidean=euclidean, euclidean_trainable=True)

        class OrthogonalGate(Parametrized):
            def __init__(self, orthogonal: Tensor):
                super.__init__(orthogonal=orthogonal, orthogonal_trainable=True)

        class UnitaryGate(Parametrized):
            def __init__(self, unitary: Array):
                super.__init__(unitary=unitary, unitary_trainable=True)

    The optimization procedure updates the value of the trainables *in-place*.

3. **Constant parameters** (bound and fixed): This class of parameters belong to the autograd
   backend but remain fixed during the optimization procedure. They are created by setting the
   trainable flag to False.

    .. code-block::

        class Gate(Parametrized):
            def __init__(self, r: float):
                super.__init__(r=r, r_trainable=False)

"""
# pylint: disable=super-init-not-called

from abc import ABC, abstractmethod

from typing import Optional, Sequence, Any
from mrmustard.math import Math
from mrmustard.typing import Tensor

math = Math()

__all__ = [
    "Parameter",
    "Trainable",
    "Symplectic",
    "Euclidean",
    "Orthogonal",
    "Constant",
    "create_parameter",
]


class Parameter(ABC):
    """Parameter abstract base class.

    This class implements common methods for :class:`Trainable` and :class:`Constant` parameters.
    """

    @abstractmethod
    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        pass

    @property
    def value(self) -> Tensor:
        """tensor value of the parameter"""
        return self._value

    @property
    def name(self) -> str:
        """name of the parameter"""
        return self._name

    @property
    def owner(self) -> str:
        """parameter owner"""
        return self._owner

    @property
    def type(self) -> str:
        """the lowercased name of the class of this parameter object"""
        return self.__class__.__name__.lower()


class Trainable(Parameter, ABC):
    """This abstract base class represent parameters that are mutable
    and can updated by the optimization procedure.

    Note that the class name of instances of ``Trainable`` are used
    to infer the optimization procedure on the :py:training: module.
    """

    @abstractmethod
    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        pass


class Symplectic(Trainable):
    """Symplectic trainable. Uses :meth:`training.parameter_update.update_symplectic`."""

    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        self._value = value_to_trainable(value, None, name)
        self._name = name
        self._owner = owner


class Euclidean(Trainable):
    """Euclidean trainable. Uses :meth:`training.parameter_update.update_euclidean`."""

    def __init__(
        self, value: Any, bounds: Optional[Sequence], name: str, owner: Optional[str] = None
    ) -> None:
        self._value = value_to_trainable(value, bounds, name)
        self._name = name
        self._owner = owner
        self.bounds = bounds


class Orthogonal(Trainable):
    """Orthogonal trainable. Uses :meth:`training.parameter_update.update_orthogonal`."""

    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        self._value = value_to_trainable(value, None, name)
        self._name = name
        self._owner = owner


class Unitary(Trainable):
    """Unitary trainable. Uses :meth:`training.parameter_update.update_unitary`."""

    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        self._value = value_to_trainable(value, None, name)
        self._name = name
        self._owner = owner


class Constant(Parameter):
    """Constant parameter. It belongs to the autograd backend but remains fixed
    during any optimization procedure
    """

    def __init__(self, value: Any, name: str, owner: Optional[str] = None) -> None:
        if math.from_backend(value) and not math.is_trainable(value):
            self._value = value
        elif type(value) in [list, int, float]:
            self._value = math.new_constant(value, name)
        else:
            self._value = math.new_constant(value, name, value.dtype)
        self._name = name
        self._owner = owner


def create_parameter(
    value: Any, name: str, is_trainable: bool = False, bounds: Optional[Sequence] = None, owner=None
) -> Trainable:
    """A factory function that returns an instance of a :class:`Trainable` given
    its arguments.

    Args:
        value: The value to be assigned to the parameter. This value
            is casted into a Tensor belonging to the backend.
        name (str): name of the parameter
        is_trainable (bool): if ``True`` the returned object is instance
            of :class:`Trainable`, else returns an instance of a :class:`Constant`
        bounds (None or Sequence): value constraints for the parameter, only applicable
            for Euclidean parameters

    Returns:
        Parameter: an instance of a :class:`Constant` or :class:`Symplectic`, :class:`Orthogonal`
            or :class:`Euclidean` trainable.
    """

    if not is_trainable:
        return Constant(value, name, owner)

    if name.startswith("symplectic"):
        return Symplectic(value, name, owner)

    if name.startswith("orthogonal"):
        return Orthogonal(value, name, owner)

    if name.startswith("unitary"):
        return Unitary(value, name, owner)

    return Euclidean(value, bounds, name, owner)


def value_to_trainable(value: Any, bounds: Optional[Sequence], name: str) -> Tensor:
    """Converts a value to a backend tensor variable if needed.

    Args:
        value: value to be casted into a tensor of the backend
        bounds (None or Sequence): value constraints for the parameter, only applicable
            for Euclidean parameters
        name (str): name of the parameter
    """
    if math.from_backend(value) and math.is_trainable(value):
        return value
    elif type(value) in [list, int, float]:
        return math.new_variable(value, bounds, name)
    else:
        return math.new_variable(value, bounds, name, value.dtype)
