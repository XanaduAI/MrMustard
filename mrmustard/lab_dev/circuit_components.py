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

# pylint: disable=super-init-not-called

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from ..physics.representations import Bargmann, Fock, Representation
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .wires import Wires

__all__ = ["CircuitComponent", "AdjointView", "DualView"]


class CircuitComponent:
    r"""
    A base class for the components (states, transformations, and measurements, or potentially
    unphysical ``wired'' objects) that can be placed in Mr Mustard's quantum circuits.

    Args:
        name: The name of this component.
        representation: A representation for this circuit component.
        modes_out_bra: The output modes on the bra side of this component.
        modes_in_bra: The input modes on the bra side of this component.
        modes_out_ket: The output modes on the ket side of this component.
        modes_in_ket: The input modes on the ket side of this component.
    """

    def __init__(
        self,
        name: str,
        representation: Optional[Bargmann | Fock] = None,
        modes_out_bra: tuple[int,...] = (),
        modes_in_bra: tuple[int,...] = (),
        modes_out_ket: tuple[int,...] = (),
        modes_in_ket: tuple[int,...] = (),
    ) -> None:
        self._name = name or ""
        self._wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        self._parameter_set = ParameterSet()
        self._representation = representation
        # handle out-of-order modes
        a,b,c,d = sorted(modes_out_bra), sorted(modes_in_bra), sorted(modes_out_ket), sorted(modes_in_ket)
        if a != sorted(a) or b != sorted(b) or c != sorted(c) or d != sorted(d):
            offsets = [0, len(a), len(a)+len(b), len(a)+len(b)+len(c)]
            perm = (tuple(np.argsort(a)) + tuple(np.argsort(b)+offsets[0])
                    + tuple(np.argsort(c)+offsets[1]) + tuple(np.argsort(d)+offsets[2]))
            if self._representation is not None:
                self._representation = self._representation.reorder(tuple(perm))

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
    def representation(self) -> Optional[Representation]:
        r"""
        A representation of this circuit component.
        """
        return self._representation

    @property
    def modes(self) -> list[int]:
        r"""
        The sorted list of modes of this component.
        """
        return sorted(self.wires.modes)

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
    def adjoint(self) -> AdjointView:
        r"""
        The ``AdjointView`` of this component.
        """
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""
        The ``DualView`` of this component.
        """
        return DualView(self)

    def light_copy(self) -> CircuitComponent:
        r"""
        Creates a copy of this component by copying every data stored in memory for
        it by reference, except for its wires, which are copied by value.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.__dict__["_wires"] = Wires(*self.wires.original.args)
        return instance

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations and wires, but not the other attributes (including name and parameter set).
        """
        return self.representation == other.representation and self.wires == other.wires

    def __matmul__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other``, without adding adjoints.
        """
        # initialized the ``Wires`` of the returned component
        wires_ret, perm = self.wires @ other.wires

        # find the indices of the wires being contracted on the bra side
        bra_modes = tuple(self.wires.bra.output.modes & other.wires.bra.input.modes)
        idx_z = self.wires.bra.output[bra_modes].indices
        idx_zconj = other.wires.bra.input[bra_modes].indices

        # find the indices of the wires being contracted on the ket side
        ket_modes = tuple(self.wires.ket.output.modes & other.wires.ket.input.modes)
        idx_z += self.wires.ket.output[ket_modes].indices
        idx_zconj += other.wires.ket.input[ket_modes].indices

        # convert Bargmann -> Fock if needed
        LEFT = self.representation
        RIGHT = other.representation
        if isinstance(LEFT, Bargmann) and isinstance(RIGHT, Fock):
            raise ValueError("Cannot contract objects with different representations.")
            # shape = [s if i in idx_z else None for i, s in enumerate(other.representation.shape)]
            # LEFT = Fock(self.fock(shape=shape), batched=False)
        if isinstance(LEFT, Fock) and isinstance(RIGHT, Bargmann):
            raise ValueError("Cannot contract objects with different representations.")
            # shape = [s if i in idx_zconj else None for i, s in enumerate(self.representation.shape)]
            # RIGHT = Fock(other.fock(shape=shape), batched=False)
        # calculate the representation of the returned component
        representation_ret = LEFT[idx_z] @ RIGHT[idx_zconj]

        # reorder the representation
        representation_ret = representation_ret.reorder(perm)

        return CircuitComponent.from_attributes("", representation_ret, wires_ret)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.
        """
        msg = f"``>>`` not supported between {self} and {other}, use ``@``."

        wires_out = self.wires.output
        wires_in = other.wires.input

        if wires_out.ket and wires_out.bra:
            if wires_in.ket and wires_in.bra:
                return self @ other
            return self @ other @ other.adjoint

        if wires_out.ket:
            if wires_in.ket and wires_in.bra:
                return self @ self.adjoint @ other
            if wires_in.ket:
                return self @ other
            raise ValueError(msg)

        if wires_out.bra:
            if wires_in.ket and wires_in.bra:
                return self @ self.adjoint @ other
            if wires_in.bra:
                return self @ other
            raise ValueError(msg)

        raise ValueError(msg)

    def __repr__(self) -> str:
        name = self.name if self.name else "None"
        return f"CircuitComponent(name={name}, modes={self.modes})"


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

    def __repr__(self) -> str:
        return repr(self._component)


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

    def __repr__(self) -> str:
        return repr(self._component)
