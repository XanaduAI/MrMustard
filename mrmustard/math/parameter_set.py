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

"""This module contains the classes to describe sets of parameters."""

from typing import Union

from .parameters import Constant, Variable

__all__ = [
    "ParameterSet",
]


class ParameterSet:
    r"""
    A set of parameters.
    """

    def __init__(self):
        self._constants: dict[str, Constant] = {}
        self._variables: dict[str, Variable] = {}

    @property
    def constants(self) -> dict[str, Constant]:
        r"""
        The constant parameters in this parameter set.
        """
        return self._constants

    @property
    def variables(self) -> dict[str, Variable]:
        r"""
        The variable parameters in this parameter set.
        """
        return self._variables

    def add_parameter(self, parameter: Union[Constant, Variable]) -> None:
        r"""
        Adds a parameter to this parameter set.

        Args:
            parameter: A constant or variable parameter.

        Raises:
            ValueError: If this parameter set already contains a parameter with the same
                name as that of the given parameter.
        """
        name = parameter.name

        if name in self.constants or name in self.variables:
            msg = f"A parameter with name ``{name}`` is already part of this parameter set."
            raise ValueError(msg)

        # updates dictionary and dynamically creates an attribute
        if isinstance(parameter, Constant):
            self.constants[name] = parameter
            self.__dict__[name] = self.constants[name]
        elif isinstance(parameter, Variable):
            self.variables[parameter.name] = parameter
            self.__dict__[name] = self.variables[name]
