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

# pylint: disable=super-init-not-called, protected-access, import-outside-toplevel
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import os
import numpy as np
from IPython.display import display, HTML
from mako.template import Template

from mrmustard.utils.typing import Scalar, ComplexTensor
from mrmustard.physics.converters import to_fock
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Constant, Variable
from mrmustard.lab_dev.wires import Wires

__all__ = ["CircuitComponent", "AdjointView", "DualView"]


class CircuitComponent:
    r"""
    A base class for the components (states, transformations, and measurements, or potentially
    unphysical ``wired'' objects) that can be placed in Mr Mustard's quantum circuits.

    Args:
        representation: A representation for this circuit component.
        modes_out_bra: The output modes on the bra side of this component.
        modes_in_bra: The input modes on the bra side of this component.
        modes_out_ket: The output modes on the ket side of this component.
        modes_in_ket: The input modes on the ket side of this component.
        name: The name of this component.
    """

    def __init__(
        self,
        representation: Optional[Bargmann | Fock] = None,
        modes_out_bra: Optional[Sequence[int]] = None,
        modes_in_bra: Optional[Sequence[int]] = None,
        modes_out_ket: Optional[Sequence[int]] = None,
        modes_in_ket: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
    ) -> None:
        modes_out_bra = modes_out_bra or ()
        modes_in_bra = modes_in_bra or ()
        modes_out_ket = modes_out_ket or ()
        modes_in_ket = modes_in_ket or ()

        self._wires = Wires(
            set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket)
        )
        self._name = name or "CC" + "".join(str(m) for m in sorted(self.wires.modes))
        self._parameter_set = ParameterSet()
        self._representation = representation

        # handle out-of-order modes
        ob = tuple(sorted(modes_out_bra))
        ib = tuple(sorted(modes_in_bra))
        ok = tuple(sorted(modes_out_ket))
        ik = tuple(sorted(modes_in_ket))
        if ob != modes_out_bra or ib != modes_in_bra or ok != modes_out_ket or ik != modes_in_ket:
            offsets = [len(ob), len(ob) + len(ib), len(ob) + len(ib) + len(ok)]
            perm = (
                tuple(np.argsort(modes_out_bra))
                + tuple(np.argsort(modes_in_bra) + offsets[0])
                + tuple(np.argsort(modes_out_ket) + offsets[1])
                + tuple(np.argsort(modes_in_ket) + offsets[2])
            )
            if self._representation:
                self._representation = self._representation.reorder(tuple(perm))

    @classmethod
    def _from_attributes(
        cls,
        representation: Representation,
        wires: Wires,
        name: Optional[str] = None,
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
            representation: A representation for this circuit component.
            wires: The wires of this component.
            name: The name of this component.

        Returns:
            A circuit component of type ``cls`` with the given attributes.
        """
        types = {"Ket", "DM", "Unitary", "Operation", "Channel", "Map"}
        for tp in cls.mro():
            if tp.__name__ in types:
                ret = tp()
                break
        else:
            ret = CircuitComponent()

        ret._name = name or tp.__name__ + "".join(str(m) for m in sorted(wires.modes))
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

    @classmethod
    def from_bargmann(
        cls,
        triple: tuple,
        modes_out_bra: Sequence[int] = (),
        modes_in_bra: Sequence[int] = (),
        modes_out_ket: Sequence[int] = (),
        modes_in_ket: Sequence[int] = (),
        name: Optional[str] = None,
    ) -> CircuitComponent:
        r"""
        Initializes a circuit component from its Bargmann representation.

        Args:
            triple: The Bargmann representation of the component.
            modes_out_bra: The output modes on the bra side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_ket: The input modes on the ket side of this component.
            name: The name of this component.

        Returns:
            A circuit component with the given Bargmann representation.
        """
        repr = Bargmann(*triple)
        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        return cls._from_attributes(repr, wires, name)

    @property
    def bargmann(self) -> tuple:
        r"""
        The Bargmann parametrization of this component, if available.
        """
        try:
            return self.representation.triple
        except AttributeError as e:
            raise AttributeError(
                f"Cannot compute triple from representation of type ``{self.representation.__class__.__qualname__}``."
            ) from e

    @classmethod
    def from_quadrature(
        cls,
        modes_out_bra: Sequence[int],
        modes_in_bra: Sequence[int],
        modes_out_ket: Sequence[int],
        modes_in_ket: Sequence[int],
        triple: tuple,
        phi: float = 0.0,
        name: Optional[str] = None,
    ) -> CircuitComponent:
        r"""Returns a circuit component from the given CV quadrature triple (A,b,c).
        It assumes that the ansatz is in the form ``c exp(1/2 x^T A x + b^T x)``.

        Args:
            modes_out_bra: The output modes on the bra side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_ket: The input modes on the ket side of this component.
            triple: The (A,b,c) triple that parametrizes the wave function.
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature, ``phi=pi/2`` to the p quadrature.
            name: The name of this component.

        Returns:
            A circuit component with the given quadrature representation.
        """
        from mrmustard.lab_dev.circuit_components_utils import BtoQ

        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        QtoB_ob = BtoQ(modes_out_bra, phi).inverse().adjoint
        QtoB_ib = BtoQ(modes_in_bra, phi).inverse().adjoint.dual
        QtoB_ok = BtoQ(modes_out_ket, phi).inverse()
        QtoB_ik = BtoQ(modes_in_ket, phi).inverse().dual
        # NOTE: the representation is Bargmann because we will use the inverse of BtoQ on the B side
        QQQQ = CircuitComponent._from_attributes(Bargmann(*triple), wires)
        BBBB = QtoB_ib @ QtoB_ik @ QQQQ @ QtoB_ok @ QtoB_ob
        return cls._from_attributes(BBBB.representation, wires, name)

    def quadrature(self, phi: float = 0.0) -> tuple | ComplexTensor:
        r"""
        The quadrature representation of this circuit component.
        """
        from mrmustard.lab_dev.circuit_components_utils import BtoQ

        BtoQ_ob = BtoQ(self.wires.output.bra.modes, phi).adjoint
        BtoQ_ib = BtoQ(self.wires.input.bra.modes, phi).adjoint.dual
        BtoQ_ok = BtoQ(self.wires.output.ket.modes, phi)
        BtoQ_ik = BtoQ(self.wires.input.ket.modes, phi).dual
        QQQQ = BtoQ_ib @ BtoQ_ik @ self @ BtoQ_ok @ BtoQ_ob
        return QQQQ.representation.data

    @property
    def representation(self) -> Representation | None:
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
        instance.__dict__["_wires"] = Wires(*self.wires.args)
        return instance

    def on(self, modes: Sequence[int]) -> CircuitComponent:
        r"""
        Creates a copy of this component that acts on the given ``modes`` instead of on the
        original modes.

        Args:
            modes: The new modes that this component acts on.

        Returns:
            The component acting on the specified modes.

        Raises:
            ValueError: If ``modes`` contains more or less modes than the original component.
        """
        modes = set(modes)

        ob = self.wires.output.bra
        ib = self.wires.input.bra
        ok = self.wires.output.ket
        ik = self.wires.input.ket
        for subset in [ob, ib, ok, ik]:
            if subset and len(subset.modes) != len(modes):
                msg = f"Expected ``{len(modes)}`` modes, found ``{len(subset.modes)}``."
                raise ValueError(msg)

        ret = self.light_copy()
        ret._wires = Wires(
            modes_out_bra=modes if ob else set(),
            modes_in_bra=modes if ib else set(),
            modes_out_ket=modes if ok else set(),
            modes_in_ket=modes if ik else set(),
        )

        return ret

    def to_fock(self, shape: Optional[Union[int, Iterable[int]]] = None) -> CircuitComponent:
        r"""
        Returns a circuit component with the same attributes as this component, but
        with ``Fock`` representation.

        Uses the :meth:`mrmustard.physics.converters.to_fock` method to convert the internal
        representation.

        .. code-block::

            >>> from mrmustard.physics.converters import to_fock
            >>> from mrmustard.lab_dev import Dgate

            >>> d = Dgate([1], x=0.1, y=0.1)
            >>> d_fock = d.to_fock(shape=3)

            >>> assert d_fock.name == d.name
            >>> assert d_fock.wires == d.wires
            >>> assert d_fock.representation == to_fock(d.representation, shape=3)

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` in the settings.
        """
        return self.__class__._from_attributes(
            to_fock(self.representation, shape=shape),
            self.wires,
            self.name,
        )

    def __add__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Implements the addition between circuit components.
        """
        if self.wires != other.wires:
            msg = "Cannot add components with different wires."
            raise ValueError(msg)
        rep = self.representation + other.representation
        name = self.name if self.name == other.name else ""
        return self._from_attributes(rep, self.wires, name)

    def __sub__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Implements the subtraction between circuit components.
        """
        if self.wires != other.wires:
            msg = "Cannot subtract components with different wires."
            raise ValueError(msg)
        rep = self.representation - other.representation
        name = self.name if self.name == other.name else ""
        return self._from_attributes(rep, self.wires, name)

    def __mul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar on the right.
        """
        return self._from_attributes(self.representation * other, self.wires, self.name)

    def __rmul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar on the left.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the division by a scalar for circuit components.
        """
        return self._from_attributes(self.representation / other, self.wires, self.name)

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations and wires, but not the other attributes (including name and parameter set).
        """
        return self.representation == other.representation and self.wires == other.wires

    def _matmul_indices(self, other: CircuitComponent) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""
        Finds the indices of the wires being contracted on the bra and ket sides of the components.
        """
        # find the indices of the wires being contracted on the bra side
        bra_modes = tuple(self.wires.bra.output.modes & other.wires.bra.input.modes)
        idx_z = self.wires.bra.output[bra_modes].indices
        idx_zconj = other.wires.bra.input[bra_modes].indices
        # find the indices of the wires being contracted on the ket side
        ket_modes = tuple(self.wires.ket.output.modes & other.wires.ket.input.modes)
        idx_z += self.wires.ket.output[ket_modes].indices
        idx_zconj += other.wires.ket.input[ket_modes].indices
        return idx_z, idx_zconj

    def __matmul__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other``, without adding adjoints.
        """
        wires_ret, perm = self.wires @ other.wires
        idx_z, idx_zconj = self._matmul_indices(other)
        rep = self.representation[idx_z] @ other.representation[idx_zconj]
        rep = rep.reorder(perm) if perm else rep
        return CircuitComponent._from_attributes(rep, wires_ret, None)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` (output of self going into input of other)
        adding the adjoints when they are missing. If this can be deduced from the wires
        of the components, the contraction is executed, otherwise an error is raised.
        """
        msg = f"``>>`` not supported between {self} and {other} because it's not clear "
        msg += "whether to add bra wires. Use ``@`` instead."

        if not self.wires.bra and not other.wires.bra:
            return self @ other  # ket side only

        if not self.wires.ket and not other.wires.ket:
            return self @ other  # bra side only

        if (self.wires.bra and self.wires.ket) and (other.wires.bra and other.wires.ket):
            return self @ other  # both sides

        if not self.wires.bra and (other.wires.bra and other.wires.ket):
            return self.adjoint @ (self @ other)  # adding self.adjoint on the bra

        if not self.wires.ket and (other.wires.bra and other.wires.ket):
            return self.adjoint @ (self @ other)  # adding self.adjoint on the ket

        if (self.wires.bra and self.wires.ket) and not other.wires.bra:
            return (self @ other) @ other.adjoint  # adding other.adjoint on the bra

        if (self.wires.bra and self.wires.ket) and not other.wires.ket:
            return (self @ other) @ other.adjoint  # adding other.adjoint on the ket

        raise ValueError(msg)

    def __repr__(self) -> str:
        return f"CircuitComponent(modes={self.modes}, name={self.name or None})"

    def _repr_html_(self):  # pragma: no cover
        temp = Template(
            filename=os.path.dirname(__file__) + "/assets/circuit_components.txt"
        )  # nosec

        wires_temp = Template(filename=os.path.dirname(__file__) + "/assets/wires.txt")  # nosec
        wires_temp_uni = wires_temp.render_unicode(wires=self.wires)
        wires_temp_uni = (
            wires_temp_uni.replace("<body>", "").replace("</body>", "").replace("h1", "h3")
        )

        rep_temp = (
            Template(filename=os.path.dirname(__file__) + "/../physics/assets/fock.txt")  # nosec
            if isinstance(self.representation, Fock)
            else Template(
                filename=os.path.dirname(__file__) + "/../physics/assets/bargmann.txt"
            )  # nosec
        )
        rep_temp_uni = rep_temp.render_unicode(rep=self.representation)
        rep_temp_uni = rep_temp_uni.replace("<body>", "").replace("</body>", "").replace("h1", "h3")

        display(HTML(temp.render(comp=self, wires=wires_temp_uni, rep=rep_temp_uni)))


class CCView(CircuitComponent):
    r"""A base class for views of circuit components.
    Args:
        component: The circuit component to take the view of.
    """

    def __init__(self, component: CircuitComponent) -> None:
        self.__dict__ = component.__dict__.copy()
        self._component = component.light_copy()

    def __getattr__(self, name):
        r"""send calls to the component"""
        return getattr(self._component, name)

    def __repr__(self) -> str:
        return repr(self._component)


class AdjointView(CCView):
    r"""
    Adjoint view of a circuit component.

    Args:
        component: The circuit component to take the view of.
    """

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
        bras = self._component.wires.bra.indices
        kets = self._component.wires.ket.indices
        return self._component.representation.reorder(kets + bras).conj()

    @property
    def wires(self):
        r"""
        The ``Wires`` in this component.
        """
        return self._component.wires.adjoint


class DualView(CCView):
    r"""
    Dual view of a circuit component.

    Args:
        component: The circuit component to take the view of.
    """

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
        ok = self._component.wires.ket.output.indices
        ik = self._component.wires.ket.input.indices
        ib = self._component.wires.bra.input.indices
        ob = self._component.wires.bra.output.indices
        return self._component.representation.reorder(ib + ob + ik + ok).conj()

    @property
    def wires(self):
        r"""
        The ``Wires`` in this component.
        """
        return self._component.wires.dual
