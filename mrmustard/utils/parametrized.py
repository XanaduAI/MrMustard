# Copyright 2021 Xanadu Quantum Technologies Inc.

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
an abstract base class for all parametrized objects.
"""

from mrmustard.types import Sequence, List
from mrmustard.math import Math
from mrmustard.utils.parameter import create_parameter, Trainable, Constant

math = Math()


class Parametrized:
    r"""A class representing all parametrized objects (gates, detectors, etc.)

    For each trainable parameter keyword arguments must be passed for the initial value ``xxx``
    (tensor), the numerical bounds ``xxx_bounds`` (float, float), whether the parameter ``xxx`` will
    be optimized ``xxx_trainable`` (bool), along with any other parameters.
    """

    def __init__(self, **kwargs):  # NOTE: only kwargs so that we can use the arg names

        owner = f"{self.__class__.__qualname__}"

        for name, value in kwargs.items():

            # filter `{name}_trainable` or `{name}_bounds`` to become fields
            # of the class as those kwargs are used to define the variables
            if "_trainable" in name or "_bounds" in name:
                continue

            # convert into parameter class
            is_trainable = kwargs.get(f"{name}_trainable", False) or math.is_trainable(value)
            bounds = kwargs.get(f"{name}_bounds", None)
            param = create_parameter(value, name, is_trainable, bounds, owner)

            # dynamically assign variable as attribute of the class
            self.__dict__[name] = param

    @property
    def trainable_parameters(self) -> Sequence[Trainable]:
        return [value for value in self.__dict__.values() if isinstance(value, Trainable)]

    @property
    def constant_parameters(self) -> List[Constant]:
        return [value for value in self.__dict__.values() if isinstance(value, Constant)]
