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

# pylint: disable=super-init-not-called, protected-access

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import numpy as np
from ..physics.converters import to_fock
from ..physics.representations import Representation
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
        name: Optional[str] = None,
        representation: Optional[Bargmann | Fock] = None,
        modes_out_bra: Optional[Sequence[int]] = (),
        modes_in_bra: Optional[Sequence[int]] = (),
        modes_out_ket: Optional[Sequence[int]] = (),
        modes_in_ket: Optional[Sequence[int]] = (),
    ) -> None:
        self._wires = Wires(
            set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket)
        )
        self._name = name or "CC" + "".join(str(m) for m in sorted(self.wires.modes))
        self._parameter_set = ParameterSet()
        self._representation = representation
        # handle out-of-order modes
        a, b, c, d = (
            sorted(modes_out_bra),
            sorted(modes_in_bra),
            sorted(modes_out_ket),
            sorted(modes_in_ket),
        )
        if a != sorted(a) or b != sorted(b) or c != sorted(c) or d != sorted(d):
            offsets = [0, len(a), len(a) + len(b), len(a) + len(b) + len(c)]
            perm = (
                tuple(np.argsort(a))
                + tuple(np.argsort(b) + offsets[0])
                + tuple(np.argsort(c) + offsets[1])
                + tuple(np.argsort(d) + offsets[2])
            )
            if self._representation is not None:
                self._representation = self._representation.reorder(tuple(perm))

    @classmethod
    def _from_attributes(
        cls, name: str, representation: Representation, wires: Wires
    ) -> CircuitComponent:
        r"""
        Initializes a circuit component from its attributes (a name, a ``Wires``,
        and a ``Representation``).

        If the Method Resolution Order (MRO) of ``cls`` contains one between ``Ket``, ``DM``,
        ``Unitary``, and ``Channel``, then the returned component is of that type. Otherwise,
        it is of type ``CircuitComponent``.

        This function needs to be used with caution, as it does not check that the attributes
        provided are consistent with the type of the returned component. If used improperly it
        may initialize, e.g., ``Ket``s with both input and output wires or ``Unitary``s with
        wires on the bra side.

        Args:
            name: The name of this component.
            representation: A representation for this circuit component.
            wires: The wires of this component.

        Returns:
            A circuit component of type ``cls`` with the given attributes.
        """
        types = {"Ket", "DM", "Unitary", "Channel"}
        for tp in cls.mro():
            if tp.__name__ in types:
                ret = tp()
                break
        else:
            ret = CircuitComponent()

        ret._name = name
        ret._representation = representation
        ret._wires = wires

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
    def representation(self) -> Representation | None:
        r"""
        A representation of this circuit component.
        """
        return self._representation

    @property
    def modes(self) -> tuple[int, ...]:
        r"""
        The sorted list of modes of this component.
        """
        return tuple(sorted(self.wires.modes))

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

    def to_fock_component(
        self, shape: Optional[Union[int, Iterable[int]]] = None
    ) -> CircuitComponent:
        r"""
        Returns a circuit component with the same attributes as this component, but
        with ``Fock`` representation.

        Uses the :meth:`mrmustard.physics.converters.to_fock` method to convert the internal
        representation.

        .. code-block::

            >>> from mrmustard.physics.converters import to_fock
            >>> from mrmustard.lab_dev import Dgate

            >>> d = Dgate([1], x=0.1, y=0.1)
            >>> d_fock = d.to_fock_component(shape=3)

            >>> assert d_fock.name == d.name
            >>> assert d_fock.wires == d.wires
            >>> assert d_fock.representation == to_fock(d.representation, shape=3)

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` in the settings.
        """
        cls = self.__class__
        return cls._from_attributes(
            self.name,
            to_fock(self.representation, shape=shape),
            self.wires,
        )

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

        # calculate the representation of the returned component
        representation_ret = self.representation[idx_z] @ other.representation[idx_zconj]

        # reorder the representation
        representation_ret = representation_ret.reorder(perm)
        return CircuitComponent._from_attributes(None, representation_ret, wires_ret)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.
        """
        msg = f"``>>`` not supported between {self} and {other}, use ``@``."

        wires_s = self.wires
        wires_o = other.wires

        if wires_s.ket and wires_s.bra:
            if wires_o.ket and wires_o.bra:
                return self @ other
            return self @ other @ other.adjoint

        if wires_s.ket:
            if wires_o.ket and wires_o.bra:
                return self @ self.adjoint @ other
            if wires_o.ket:
                return self @ other
            raise ValueError(msg)

        if wires_s.bra:
            if wires_o.ket and wires_o.bra:
                return self @ self.adjoint @ other
            if wires_o.bra:
                return self @ other
            raise ValueError(msg)

        raise ValueError(msg)

    def __repr__(self) -> str:
        return f"CircuitComponent(name={self.name}, modes={self.modes})"


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
