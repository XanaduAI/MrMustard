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

"""This module contains the :class:`.Parametrized` class which acts as
an abstract base class for all parametrized objects. Arguments of the
class constructor generate a backend Tensor and are assigned to fields
of the class.
"""

from typing import Any, Generator, List, Sequence, Tuple

from mrmustard.math import Math
from mrmustard.training.parameter import (
    Constant,
    Parameter,
    Trainable,
    create_parameter,
)
from mrmustard.types import Tensor

math = Math()

__all__ = ["Parametrized"]


class Parametrized:
    r"""A class representing all parametrized objects (gates, detectors, etc.). This class
    creates backend tensors out of the arguments of its class constructor and assigns it
    to fields of the parent class. The main role of this class is classifying and providing
    methods to keep track of trainable parameters.

    For each trainable parameter keyword arguments must be passed for the initial value ``xxx``
    (tensor), the numerical bounds ``xxx_bounds`` (float, float), whether the parameter ``xxx`` will
    be optimized ``xxx_trainable`` (bool), along with any other parameters.
    """

    def __init__(self, **kwargs):  # NOTE: only kwargs so that we can use the arg names
        owner = f"{self.__class__.__qualname__}"
        self.param_names = []  # list of parameter names to preserve order
        for name, value in kwargs.items():
            # filter out `{name}_trainable` or `{name}_bounds`` to become fields
            # of the class as those kwargs are used to define the variables
            if "_trainable" in name or "_bounds" in name:
                continue

            # convert into parameter class
            is_trainable = kwargs.get(f"{name}_trainable", False) or math.is_trainable(value)
            bounds = kwargs.get(f"{name}_bounds", None)
            param = create_parameter(value, name, is_trainable, bounds, owner)

            # dynamically assign parameter as attribute of the class
            self.__dict__[name] = param
            self.param_names.append(name)

    def param_string(self, decimals: int) -> str:
        r"""Returns a string representation of the parameter values, separated by commas and rounded
        to the specified number of decimals. It includes only the parameters that are not arrays
        and not the number of modes, or other parameters that are not in principle trainable.

        Args:
            decimals (int): number of decimals to round to

        Returns:
            str: string representation of the parameter values
        """
        string = ""
        for _, value in self.kw_parameters:
            if math.asnumpy(value).ndim == 0:  # don't show arrays
                string += f"{math.asnumpy(value):.{decimals}g}, "
        return string.rstrip(", ")

    @property
    def kw_parameters(self) -> Tuple[Tuple[str, Tensor]]:
        r"""Return a list of parameters within the Parametrized object
        if they have been passed as keyword arguments to the class constructor.
        """
        return tuple((name, getattr(self, name).value) for name in self.param_names)

    @property
    def trainable_parameters(self) -> Sequence[Trainable]:
        """Return a list of trainable parameters within the Parametrized object
        by recursively traversing the object's fields
        """
        return list(_traverse_parametrized(self.__dict__.values(), Trainable))

    @property
    def constant_parameters(self) -> List[Constant]:
        """Return a list of constant parameters within the Parametrized object
        by recursively traversing the object's fields
        """
        return list(_traverse_parametrized(self.__dict__.values(), Constant))


def _traverse_parametrized(object_: Any, extract_type: Parameter) -> Generator:
    """This private method traverses recursively all the object's attributes for objects
    present in ``iterable`` which are instances of ``parameter_type`` or ``Parametrized``
    returning a generator with objects of type ``extract_type``.
    """

    for obj in object_:
        if isinstance(
            obj, (List, Tuple)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            yield from _traverse_parametrized(obj, extract_type)
        elif isinstance(obj, Parametrized):
            yield from _traverse_parametrized(obj.__dict__.values(), extract_type)
        elif isinstance(obj, extract_type):
            yield obj
