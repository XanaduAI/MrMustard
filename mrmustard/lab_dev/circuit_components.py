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

"""
A base class for the components of quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Union
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .wires import Wired


class CircuitComponent:
    r"""
    A base class for the components (states, transformations, and measurements)
    of quantum circuits.

    Arguments:
        wires: The wires of this ``CircuitComponent``.
    """

    def __init__(
        self,
        name: str,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ) -> None:
        self._name = name
        self._wires = Wired(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
        self._parameter_set = ParameterSet()

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to a circuit object.

        Arguments:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def name(self):
        r"""
        The name of this component.
        """
        return self._name

    @property
    def parameter_set(self):
        r"""
        The set of parameters for this transformation.
        """
        return self._parameter_set

    @property
    def wires(self) -> Wired:
        r"""
        The wires of this ``CircuitComponent``.
        """
        return self._wires

    def light_copy(self) -> CircuitComponent:
        r""" """
        instance = super().__new__(self.__class__)
        instance.__dict__ = {k: v for k, v in self.__dict__.items() if k != "wires"}
        instance.__dict__["_wires"] = self.wires.new()
        return instance
