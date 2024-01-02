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

from ..physics.representations import Bargmann, Representation
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from ..utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Mode
from .wires import Wires

__all__ = ["CircuitComponent"]


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
        representation: Representation,
        modes_in_ket: Sequence[Mode] = [],
        modes_out_ket: Sequence[Mode] = [],
        modes_in_bra: Sequence[Mode] = [],
        modes_out_bra: Sequence[Mode] = [],
    ) -> None:
        self.name = name
        self.wires = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)
        self.parameter_set = ParameterSet()
        self._representation = representation

    @property
    def representation(self) -> Representation:
        r"""
        The representation of this component.
        """
        return self._representation
    
    @property
    def modes(self) -> Sequence[Mode]:
        r"""
        The modes of this component.
        """
        return self.wires.modes

    @classmethod
    def from_ABC(
        cls,
        name: str,
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
        c: Batch[ComplexTensor],
        modes_in_ket: Sequence[Mode] = [],
        modes_out_ket: Sequence[Mode] = [],
        modes_in_bra: Sequence[Mode] = [],
        modes_out_bra: Sequence[Mode] = [],
    ):
        r"""
        Initializes a circuit component from Bargmann's A, B, and c.
        """
        return cls(
            name,  Bargmann(A, b, c), modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra,
        )

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to this circuit component.

        Arguments:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a "copy" of this component where each attribute is a reference to the original
        except for a new set of wires.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.wires = self.wires.copy()
        return instance
