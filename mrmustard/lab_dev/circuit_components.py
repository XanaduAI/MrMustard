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

from typing import Optional, Sequence, Union
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from ..utils.typing import Mode
from .wires import Wires

__all__ = ["CircuitComponent",]


class CircuitComponent:
    r"""
    A base class for the components (states, transformations, and measurements)
    of quantum circuits.

    Arguments:
        name: The name of this component.
        modes_in_ket: The input modes on the ket side.
        modes_out_ket: The output modes on the ket side.
        modes_in_bra: The input modes on the bra side.
        modes_out_bra: The output modes on the bra side.
    """

    def __init__(
        self,
        name: str,
        modes_in_ket: Optional[Sequence[Mode]] = None,
        modes_out_ket: Optional[Sequence[Mode]] = None,
        modes_in_bra: Optional[Sequence[Mode]] = None,
        modes_out_bra: Optional[Sequence[Mode]] = None,
    ) -> None:
        self._name = name
        self._wires = Wires(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
        self._parameter_set = ParameterSet()

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to this circuit component.

        Arguments:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def modes(self) -> set(Mode):
        r"""
        A set with all the modes in this component.
        """
        return self.wires.modes

    @property
    def name(self) -> str:
        r"""
        The name of this component.
        """
        return self._name

    @property
    def parameter_set(self) -> ParameterSet:
        r"""
        The set of parameters characterizing this component.
        """
        return self._parameter_set

    @property
    def wires(self) -> Wires:
        r"""
        The ``Wires`` in this component.
        """
        return self._wires

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a copy of this component by copying every data stored in memory for
        it by reference, except for its wires, which are copied by value.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = {k: v for k, v in self.__dict__.items() if k != "wires"}
        instance.__dict__["_wires"] = self.wires.new()
        return instance

    def __getitem__(self, idx: Union[Mode, Sequence[Mode]]):
        r"""
        Returns a slice of this component for the given modes.
        """
        ret = CircuitComponent(self.name)
        ret._wires = self._wires[idx]
        ret._parameter_set = self.parameter_set
        return ret
