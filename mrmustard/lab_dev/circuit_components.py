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

from ..utils.typing import Scalar
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
        representation: Optional[Representation] = None,
        modes_out_bra: Optional[Sequence[int]] = None,
        modes_in_bra: Optional[Sequence[int]] = None,
        modes_out_ket: Optional[Sequence[int]] = None,
        modes_in_ket: Optional[Sequence[int]] = None,
    ) -> None:
        self._name = name or ""
        self._wires = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)
        self._parameter_set = ParameterSet()
        self._representation = representation

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
    def n_modes(self) -> list[int]:
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
        instance.__dict__["_wires"] = self.wires.copy()
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

    def __add__(self, other: CircuitComponent):
        r"""
        Implements the addition between circuit components.
        """
        if self.wires != other.wires:
            msg = "Cannot add components with different wires."
            raise ValueError(msg)
        rep = self.representation + other.representation
        return self._from_attributes(self.name, rep, self.wires)

    def __mul__(self, other: Union[CircuitComponent, Scalar]):
        r"""
        Implements the multiplication with a scalar or another component for circuit components.
        """
        if isinstance(other, CircuitComponent):
            rep = self.representation * other.representation
            wires = self.wires + other.wires
            return self._from_attributes(self.name, rep, wires)
        return self._from_attributes(self.name, other * self.representation, self.wires)

    def __rmul__(self, other: Representation | Scalar) -> Representation:
        r"""
        Implements the multiplication for circuit components with ``other`` on the right.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Scalar):
        r"""
        Implements the division by a scalar for circuit components.
        """
        return self._from_attributes(self.name, self.representation / other, self.wires)

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
        wires_ret = self.wires @ other.wires

        # find the indices of the wires being contracted on the bra side
        bra_modes = set(self.wires.bra.output.modes).intersection(other.wires.bra.input.modes)
        idx_z = self.wires[bra_modes].bra.output.indices
        idx_zconj = other.wires[bra_modes].bra.input.indices

        # find the indices of the wires being contracted on the ket side
        ket_modes = set(self.wires.ket.output.modes).intersection(other.wires.ket.input.modes)
        idx_z += self.wires[ket_modes].ket.output.indices
        idx_zconj += other.wires[ket_modes].ket.input.indices

        # calculate the representation of the returned component
        representation_ret = self.representation[idx_z] @ other.representation[idx_zconj]

        # reorder the representation
        contracted_idx = [self.wires.ids[i] for i in range(len(self.wires.ids)) if i not in idx_z]
        contracted_idx += [
            other.wires.ids[i] for i in range(len(other.wires.ids)) if i not in idx_zconj
        ]
        order = [contracted_idx.index(id) for id in wires_ret.ids]
        representation_ret = representation_ret.reorder(order) if order else representation_ret

        return CircuitComponent._from_attributes("", representation_ret, wires_ret)

    def __lshift__(self, other: CircuitComponent):
        r"""
        Projects this component onto another component by computing
        ``(other >> self.dual).dual``.
        """
        return (other >> self.dual).dual

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.
        """
        msg = f"``__rshift__`` not supported between {self} and {other}, use ``__matmul__``."

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
