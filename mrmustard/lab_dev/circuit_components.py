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

from ..physics.representations import Bargmann, Fock
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .wires import Wires


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
        representation: Optional[Bargmann | Fock] = None,
        modes_out_bra: Optional[Sequence[int]] = None,
        modes_in_bra: Optional[Sequence[int]] = None,
        modes_out_ket: Optional[Sequence[int]] = None,
        modes_in_ket: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> None:
        # TODO: Add validation to check that wires and representation are compatible (e.g.,
        # that wires have as many modes as has the representation).
        self._name = name
        self._wires = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)
        self._parameter_set = ParameterSet()
        self._representation = representation

    @classmethod
    def from_attributes(
        cls,
        name: str,
        wires: Wires,
        representation: Bargmann | Fock,
    ):
        r"""
        Initializes a circuit component from its attributes (a name, a ``Wires``,
        and a ``Representation`` like ``Bargmann`` or ``Fock``).
        """
        ret = CircuitComponent(name)
        ret._wires = wires
        ret._representation = representation
        return ret

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to this circuit component.

        Arguments:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    def __eq__(self, other) -> bool:
        r"""Returns whether the states are equal. Modes and all."""
        return self.representation == other.representation and self.wires == other.wires

    def __lshift__(self, other: CircuitComponent) -> CircuitComponent | complex:
        r"""dual of __rshift__"""
        return (other.dual >> self.dual).dual

    def __add__(self, other: CircuitComponent):
        r"""Implements addition of states. The meaning will be superposition in Hilbert space
        if the states are Kets and mixture in the case of Density Matrices."""
        if self.modes != other.modes:
            raise ValueError(f"Can't add states on different modes (got {self.modes} and {other.modes})")
        return self.__class__(self.representation + other.representation, *self.wires._args())

    def __rmul__(self, other: Union[int, float, complex]):
        r"""Implements multiplication from the left if the object on the left
        does not implement __mul__ for the type of self.

        E.g., ``0.5 * psi``.
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(other * self.representation, modes=self.modes)

    def __mul__(self, other):
        r"""Implements multiplication of two objects."""
        if isinstance(other, (int, float, complex)):
            return other * self
        modes = [m for m in self.modes if m not in other.modes] + [
            m for m in other.modes if m not in self.modes
        ]
        return self.__class__(self.representation * other.representation, modes=modes)

    def __truediv__(self, other: Union[int, float, complex]):
        r"""Implements division by a scalar.

        E.g. ``psi / 0.5``
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(self.representation / other, modes=self.modes)

    def bargmann(self):
        r"""Returns the bargmann parameters if available. Otherwise, the representation
        object raises an AttributeError.

        Returns:
            tuple[np.ndarray, np.ndarray, complex]: the bargmann parameters
        """
        if isinstance(self.representation, Bargmann):
            return self.representation.A, self.representation.b, self.representation.c
        raise AttributeError("Cannot convert to Bargmann representation.")


    @property
    def representation(self) -> Optional[Bargmann | Fock]:
        r"""
        A representation of this circuit component.
        """
        return self._representation

    @property
    def modes(self) -> list(Mode):
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

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the adjoint of it, obtained by taking the
        conjugate of the representation and switching ket and bra wires.
        """
        ret = self.light_copy()
        name = ret.name + "_adj"
        wires = ret.wires.adjoint
        representation = ret.representation.conj()
        return CircuitComponent.from_attributes(name, wires, representation)

    @property
    def dual(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the dual of it, obtained by taking the
        conjugate of the representation and switching input and output wires.
        """
        ret = self.light_copy()
        ret._name += "_dual"
        ret._wires = ret.wires.dual
        ret._representation = ret.representation.conj()
        return ret

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a copy of this component by copying every data stored in memory for
        it by reference, except for its wires, which are copied by value.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.__dict__["_wires"] = self.wires.copy() # this assigns new ids, which is what we want
        return instance
    
    def __matmul__(self, other: CircuitComponent) -> CircuitComponent:
        r"""Contracts self and other (as the would in a circuit), but without adding
        eventual missing adjoints."""
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        new_wires = self.wires >> other.wires
        idx_z = self.wires[common].output.indices
        idx_zconj = other.wires[common].input.indices
        # 3) convert bargmann -> fock if needed
        LEFT = self.representation
        RIGHT = other.representation
        if isinstance(self.representation, Bargmann) and isinstance(other.representation, Fock):
            shape = [s if i in idx_z else None for i, s in enumerate(other.representation.shape)]
            LEFT = Fock(self.fock(shape=shape), batched=False)  # assumes fock method is implemented
            RIGHT = other.representation
        if isinstance(self.representation, Fock) and isinstance(other.representation, Bargmann):
            shape = [s if i in idx_zconj else None for i, s in enumerate(self.representation.shape)]
            LEFT = self.representation
            RIGHT = Fock(other.fock(shape=shape), batched=False)  # assumes fock method is implemented
        # 4) apply low-level rshift
        new_repr = LEFT[idx_z] @ RIGHT[idx_zconj]
        before_ids = [id for id in self.wires.ids + other.wires.ids if id not in self.wires[common].output.ids]
        order = [before_ids.index(id) for id in new_wires.ids]
        new_repr = new_repr.reorder(order)
        return CircuitComponent.from_attributes(self.name + " @ " + other.name, new_wires, new_repr)



def connect(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of circuit components and connects their wires.

    In particular, it generates a list of light copies of the given components, then it modifies
    the wires' ``id``s so that connected wires have the same ``id``. It returns the list of light
    copies, leaving the input list unchanged.
    """
    ret = [component.light_copy() for component in components]

    output_ket = {m: None for c in components for m in c.modes}
    output_bra = {m: None for c in components for m in c.modes}

    for component in ret:
        for mode in component.modes:
            if component.wires[mode].ket.ids:
                if output_ket[mode]:
                    component.wires[mode].input.ket.ids = output_ket[mode].output.ket.ids
                output_ket[mode] = component.wires[mode]

            if component.wires[mode].bra.ids:
                if output_bra[mode]:
                    component.wires[mode].input.bra.ids = output_bra[mode].output.bra.ids
                output_bra[mode] = component.wires[mode]
    return ret


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
