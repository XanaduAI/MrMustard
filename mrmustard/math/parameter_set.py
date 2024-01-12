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

from typing import Sequence, Union

from mrmustard.math.backend_manager import BackendManager

from .parameters import Constant, Variable

math = BackendManager()

__all__ = [
    "ParameterSet",
]


class ParameterSet:
    r"""
    A set of parameters.

    ``ParameterSet`` can store both constant and variable parameters. It provides fast access to
    both classes of parameters, as well as to their names.

    .. code::

      const1 = Constant(1.2345, "const1")
      const2 = Constant(2.3456, "const2")
      var1 = Variable(3.4567, "var1")

      ps = ParameterSet()
      ps.add_parameter(const1)
      ps.add_parameter(const2)
      ps.add_parameter(var1)

      ps.names  # returns `["const1", "const2", "var1"]`
      ps.constants  # returns `{"const1": const1, "const2": const2}`
      ps.variable  # returns `{"var1": var1}`
    """

    def __init__(self):
        self._names: list[str] = []
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

    @property
    def names(self) -> Sequence[str]:
        r"""
        The names of all the parameters in this parameter set, in the order in which they
        were added.
        """
        return self._names

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

        if name in self.names:
            msg = f"A parameter with name ``{name}`` is already part of this parameter set."
            raise ValueError(msg)
        self._names.append(name)

        # updates dictionary and dynamically creates an attribute
        if isinstance(parameter, Constant):
            self.constants[name] = parameter
            self.__dict__[name] = self.constants[name]
        elif isinstance(parameter, Variable):
            self.variables[parameter.name] = parameter
            self.__dict__[name] = self.variables[name]

    def tagged_variables(self, tag: str) -> dict[str, Variable]:
        r"""
        Returns a dictionary whose keys are tagged names of the variables in this parameter set, and whose
        values are the variables in this parameter set. Tagging is done by prepending the string ``f"{tag}"/``
        to variables' original names.
        """
        ret = {}
        for k, v in self.variables.items():
            ret[f"{tag}/{k}"] = v
        return ret

    def to_string(self, decimals: int) -> str:
        r"""
        Returns a string representation of the parameter values, separated by commas and rounded
        to the specified number of decimals.

        Args:
            decimals (int): number of decimals to round to

        Returns:
            str: string representation of the parameter values
        """
        strings = []
        for name in self.names:
            param = self.constants.get(name) or self.variables.get(name)
            if len(param.value.shape) == 0:  # don't show arrays
                string = str(math.asnumpy(math.round(param.value, decimals)))
            else:
                string = f"{name}"
            strings.append(string)
        return ", ".join(strings)
