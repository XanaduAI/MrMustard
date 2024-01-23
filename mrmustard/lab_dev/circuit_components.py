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

__all__ = [
    "CircuitComponent",
]


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
        representation: A representation of this circuit component.
    """

    def __init__(
        self,
        name: str,
        modes_out_bra: Optional[Sequence[Mode]] = None,
        modes_in_bra: Optional[Sequence[Mode]] = None,
        modes_out_ket: Optional[Sequence[Mode]] = None,
        modes_in_ket: Optional[Sequence[Mode]] = None,
        representation: Optional[Representation] = None,
    ) -> None:
        # TODO: Add validation to check that wires and representation are compatible (e.g.,
        # that wires have as many modes as has the representation).
        self._name = name
        self._wires = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)
        self._parameter_set = ParameterSet()
        self._representation = representation

    @classmethod
    def from_ABC(
        cls,
        name: str,
        A: Batch[ComplexMatrix],
        B: Batch[ComplexVector],
        c: Batch[ComplexTensor],
        modes_in_ket: Optional[Sequence[Mode]] = None,
        modes_out_ket: Optional[Sequence[Mode]] = None,
        modes_in_bra: Optional[Sequence[Mode]] = None,
        modes_out_bra: Optional[Sequence[Mode]] = None,
    ):
        r"""
        Initializes a circuit component from Bargmann's A, B, and c.
        """
        ret = CircuitComponent(
            name, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra, Bargmann(A, B, c)
        )
        return ret

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to this circuit component.

        Arguments:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def representation(self) -> Representation:
        r"""
        A representation of this circuit component.
        """
        return self._representation

    @property
    def modes(self) -> set(Mode):
        r"""
        A set with all the modes in this component.
        """
        return self.wires.modes
    
    @property
    def num_modes(self) -> int:
        r"""
        The number of modes in this component.
        """
        return len(self.modes)

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

    def adjoint(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the adjoint of it, obtained by switching
        ket and bra wires.
        """
        ret = self.light_copy()
        ret._name += "_adj"
        ret._wires = ret.wires.adjoint()
        return ret
    
    def dual(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the dual of it, obtained by switching
        in and out wires.
        """
        ret = self.light_copy()
        ret._name += "_dual"
        ret._wires = ret.wires.dual()
        return ret

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a copy of this component by copying every data stored in memory for
        it by reference, except for its wires, which are copied by value.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = {k: v for k, v in self.__dict__.items() if k != "wires"}
        instance.__dict__["_wires"] = self.wires.copy()
        return instance

    def __getitem__(self, idx: Union[Mode, Sequence[Mode]]):
        r"""
        Returns a slice of this component for the given modes.
        """
        ret = self.light_copy()
        ret._wires = self._wires[idx]
        ret._parameter_set = self.parameter_set
        return ret


# def connect(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
#     r"""
#     Takes as input a sequence of circuit components and connects their wires.

#     In particular, it generates a list of light copies of the given components, then it modifies
#     the wires' ``id``s so that connected wires have the same ``id``. It returns the list of light
#     copies, leaving the input list unchanged.
#     """
#     ret = [component.light_copy() for component in components]

#     output_ket = {m: None for c in components for m in c.modes}
#     output_bra = {m: None for c in components for m in c.modes}

#     for component in ret:
#         for mode in component.modes:
#             if component.wires[mode].ket:
#                 if output_ket[mode]:
#                     component.wires[mode].input.ket.ids = output_ket[mode].output.ket.ids
#                 output_ket[mode] = component.wires[mode]

#             if component.wires[mode].bra.ids:
#                 if output_bra[mode]:
#                     component.wires[mode].input.bra.ids = output_bra[mode].output.bra.ids
#                 output_bra[mode] = component.wires[mode]
#     return ret


def add_bra(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of circuit components and adds the adjoint of every component that
    has no wires on the bra side.

    It works on light copies of the given components, so the input list is not mutatd.

    Args:
        components: A sequence of circuit components.

    Returns:
        The new components.
    """
    ret = []

    for component in components:
        ret.append(component.light_copy())
        if not component.wires.bra:
            ret.append(component.adjoint())

    return ret
