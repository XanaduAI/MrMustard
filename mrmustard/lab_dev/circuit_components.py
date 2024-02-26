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

from ..physics.representations import Bargmann, Fock, Representation
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from ..utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector
from .wires import Wires

__all__ = ["CircuitComponent", "AdjointView", "DualView", "add_bra", "connect"]


class CircuitComponent:
    r"""
    A base class for the components (states, transformations, and measurements, or potentially
    unphysical ``wired'' objects) that can be placed in Mr Mustard's quantum circuits.

    Args:
        name: The name of this component.
        representation: A representation for this circuit component.
        modes_in_ket: The input modes on the ket side of this component.
        modes_out_ket: The output modes on the ket side of this component.
        modes_in_bra: The input modes on the bra side of this component.
        modes_out_bra: The output modes on the bra side of this component.
    """

    def __init__(
        self,
        name: str,
        representation: Optional[Representation] = None,
        modes_out_bra: Optional[Sequence[int]] = None,
        modes_in_bra: Optional[Sequence[int]] = None,
        modes_out_ket: Optional[Sequence[int]] = None,
        modes_in_ket: Optional[Sequence[int]] = None,
    ) -> None:
        # TODO: Add validation to check that wires and representation are compatible (e.g.,
        # that wires have as many modes as has the representation).
        self._name = name
        self._wires = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)
        self._parameter_set = ParameterSet()
        self._representation = representation

    @classmethod
    def from_bargmann(
        cls,
        name: str,
        Abc: Union[
            tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]], Bargmann
        ],
        modes_in_ket: Optional[Sequence[int]] = None,
        modes_out_ket: Optional[Sequence[int]] = None,
        modes_in_bra: Optional[Sequence[int]] = None,
        modes_out_bra: Optional[Sequence[int]] = None,
    ):
        r"""
        Initializes a circuit component from a Bargmann ``Representation``.

        Args:
            name: The name of this component.
            Abc: An ``(A, b, c)`` triple or a Bargmann ``Representation`` for this circuit component.
            modes_in_ket: The input modes on the ket side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_bra: The output modes on the bra side of this component.

        Returns:
            A circuit component.
        """
        representation = Abc if isinstance(Abc, Bargmann) else Bargmann(*Abc)
        return CircuitComponent(
            name, representation, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra
        )

    @classmethod
    def from_attributes(
        cls,
        name: str,
        representation: Representation,
        wires: Wires,
    ):
        r"""
        Initializes a circuit component from its attributes (a name, a ``Wires``,
        and a ``Representation``).

        Args:
            name: The name of this component.
            representation: A representation for this circuit component.
            wires: The wires of this component.

        Returns:
            A circuit component.
        """
        ret = CircuitComponent(name)
        ret._wires = wires
        ret._representation = representation
        return ret

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to this circuit component.

        Args:
            parameter: The parameter to add.

        Raises:
            ValueError: If the length of the given parameter is incompatible with the number
                of modes.
        """
        if parameter.value.shape != ():
            if len(parameter.value) != 1 and len(parameter.value) != len(self.modes):
                msg = f"Length of ``{parameter.name}`` must be 1 or {len(self.modes)}."
                raise ValueError(msg)
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def representation(self) -> Representation:
        r"""
        A representation of this circuit component.
        """
        return self._representation

    @property
    def modes(self) -> list[int]:
        r"""
        The sorted list of modes of this component.
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
        The wires of this component.
        """
        return self._wires

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the adjoint of it, obtained by taking the
        conjugate of the representation and switching ket and bra wires.
        """
        return AdjointView(self)

    @property
    def dual(self) -> CircuitComponent:
        r"""
        Light-copies this component, then returns the dual of it, obtained by taking the
        conjugate of the representation and switching input and output wires.
        """
        return DualView(self)

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a copy of this component by copying every data stored in memory for
        it by reference, except for its wires, which are copied by value.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.__dict__["_wires"] = self.wires.copy()
        return instance

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations and wires, but not the other attributes (including name and parameter set).
        """
        return self.representation == other.representation and self.wires == other.wires

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        r"""
        Returns a slice of this component for the given modes.
        """
        ret = self.light_copy()
        ret._wires = self._wires[idx]
        ret._parameter_set = self.parameter_set
        return ret

    def __matmul__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other``, without adding adjoints.
        """
        # initialized the ``Wires`` of the returned component
        wires_ret = self.wires @ other.wires

        # find the indices of the wires being contracted on the bra side
        bra_modes = set(self.wires.bra.output.modes).intersection(other.wires.bra.input.modes)
        idx_z = self.wires[bra_modes].bra.output.indices
        idx_zconj = other.wires[bra_modes].bra.input.indices

        # find the indices of the wires being contracted on the ket side
        ket_modes = set(self.wires.ket.output.modes).intersection(other.wires.ket.input.modes)
        idx_z += self.wires[ket_modes].ket.output.indices
        idx_zconj += other.wires[ket_modes].ket.input.indices

        # convert Bargmann -> Fock if needed
        LEFT = self.representation
        RIGHT = other.representation
        if isinstance(LEFT, Bargmann) and isinstance(RIGHT, Fock):
            raise ValueError("Cannot contract objects with different representations.")
            # shape = [s if i in idx_z else None for i, s in enumerate(other.representation.shape)]
            # LEFT = Fock(self.fock(shape=shape), batched=False)
        elif isinstance(LEFT, Fock) and isinstance(RIGHT, Bargmann):
            raise ValueError("Cannot contract objects with different representations.")
            # shape = [s if i in idx_zconj else None for i, s in enumerate(self.representation.shape)]
            # RIGHT = Fock(other.fock(shape=shape), batched=False)

        # calculate the representation of the returned component
        representation_ret = LEFT[idx_z] @ RIGHT[idx_zconj]

        # reorder the representation
        contracted_idx = [self.wires.ids[i] for i in range(len(self.wires.ids)) if i not in idx_z]
        contracted_idx += [
            other.wires.ids[i] for i in range(len(other.wires.ids)) if i not in idx_zconj
        ]
        order = [contracted_idx.index(id) for id in wires_ret.ids]
        representation_ret = representation_ret.reorder(order)

        return CircuitComponent.from_attributes("", wires_ret, representation_ret)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.
        """
        ret = self @ other

        if not self.wires.bra:
            if not other.wires.bra:
                # self has ket, other has ket
                return ret
            elif not other.wires.ket:
                # self has ket, other has bra
                return ret @ ret.adjoint
            # self has ket, other has ket and bra
            return self.adjoint @ ret
        elif not self.wires.ket:
            if not other.wires.bra:
                # self has bra, other has ket
                return ret @ ret.adjoint
            elif not other.wires.ket:
                # self has bra, other has bra
                return ret
            # self has bra, other has ket and bra
            return self.adjoint @ ret
        if not other.wires.bra or not other.wires.ket:
            # self has ket and bra, other has ket or bra
            return ret @ other.adjoint
        # self has ket and bra, other has ket and bra
        return ret

    def __repr__(self) -> str:
        return f"CircuitComponent(name = {self.name}, modes = {self.modes})"


class AdjointView(CircuitComponent):
    r"""
    Adjoint view of a circuit component.

    Args:
        component: The circuit component to take the view of.
    """

    def __init__(self, component: CircuitComponent) -> None:
        self.__dict__ = component.light_copy().__dict__.copy()
        self._component = component.light_copy()

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        Returns a light-copy of the component that was used to generate the view.
        """
        return self._component.light_copy()

    @property
    def representation(self):
        r"""
        A representation of this circuit component.
        """
        return self._component.representation.conj()

    @property
    def wires(self):
        r"""
        The ``Wires`` in this component.
        """
        return self._component.wires.adjoint


class DualView(CircuitComponent):
    r"""
    Dual view of a circuit component.

    Args:
        component: The circuit component to take the view of.
    """

    def __init__(self, component: CircuitComponent) -> None:
        self.__dict__ = component.__dict__.copy()
        self._component = component.light_copy()

    @property
    def dual(self) -> CircuitComponent:
        r"""
        Returns a light-copy of the component that was used to generate the view.
        """
        return self._component.light_copy()

    @property
    def representation(self):
        r"""
        A representation of this circuit component.
        """
        return self._component.representation.conj()

    @property
    def wires(self):
        r"""
        The ``Wires`` in this component.
        """
        return self._component.wires.dual


def add_bra(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of circuit components and adds the adjoint of every component that
    has no wires on the bra side.

    It works on light copies of the given components, so the input list is not mutated.

    Args:
        components: The circuit components to add bras to.

    Returns:
        The connected components, light-copied.
    """
    ret = []

    for component in components:
        component_cp = component.light_copy()
        if not component_cp.wires.bra:
            ret.append(component_cp @ component_cp.adjoint)
        else:
            ret.append(component_cp)
    return ret


def connect(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of circuit components and connects their wires.

    In particular, it generates a list of light copies of the given components, then it modifies
    the wires' ``id``s so that connected wires have the same ``id``. It returns the list of light
    copies, leaving the input list unchanged.

    Args:
        components: The circuit components to connect.

    Returns:
        The connected components, light-copied.
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
