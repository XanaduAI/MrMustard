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
import numbers

import os
import numpy as np
from IPython.display import display, HTML
from mako.template import Template

from mrmustard import math, settings
from mrmustard.utils.typing import Scalar, ComplexTensor
from mrmustard.physics.converters import to_fock
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Constant, Variable
from mrmustard.lab_dev.wires import Wires

__all__ = ["CircuitComponent", "AdjointView", "DualView"]


class CircuitComponent:
    r"""
    A base class for the circuit components (states, transformations, measurements,
    and any component made by combining CircuitComponents). CircuitComponents are
    defined by their ``representation`` and ``wires`` attributes. See the :class:`Wires`
    and :class:`Representation` classes (and their subclasses) for more details.

    Args:
        representation: A representation for this circuit component.
        modes_out_bra: The output modes on the bra side of this component.
        modes_in_bra: The input modes on the bra side of this component.
        modes_out_ket: The output modes on the ket side of this component.
        modes_in_ket: The input modes on the ket side of this component.
        name: The name of this component.
    """

    short_name = "CC"

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
        self._name = name  # or "CC" + "".join(str(m) for m in sorted(self.wires.modes))
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
        Initializes a circuit component from a ``Representation``, a set of ``Wires``, a name.
        It differs from the __init__ in that it takes a set of wires directly.
        Note there are deliberately no checks to ensure types and wires are compatible
        in the standard way (e.g. one could pass a representation for a single mode ket
        and wires for a two-mode one).

        The return type is the closest parent among the types ``Ket``, ``DM``, ``Unitary``,
        ``Operation``, ``Channel``, and ``Map``. This is to ensure the right properties
        are used when calling methods on the returned object, e.g. when adding two
        coherent states we don't get a generic ``CircuitComponent`` but a ``Ket``:

        .. code-block::
            >>> from mrmustard.lab_dev import Coherent, Ket
            >>> cat = Coherent(modes=[0], x=2.0) + Coherent(modes=[0], x=-2.0)
            >>> assert isinstance(cat, Ket)

        Args:
            representation: A representation for this circuit component.
            wires: The wires of this component.
            name: The name for this component (optional).

        Returns:
            A circuit component with the given attributes.
        """
        types = {"Ket", "DM", "Unitary", "Operation", "Channel", "Map"}
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
        Adds a parameter to this circuit component and makes it accessible as an attribute.

        Args:
            parameter: The parameter to add.

        Raises:
            ValueError: If the the given parameter is incompatible with the number
                of modes (e.g. for parallel gates).
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
        Initializes a ``CircuitComponent`` object from its Bargmann (A,b,c) parametrization.

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
        r"""The Bargmann parametrization of this component, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this component is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`

        .. code-block:: pycon

            >>> from mrmustard.lab_dev import CircuitComponent, Coherent
            >>> coh = Coherent(modes=[0], x=1.0)
            >>> coh_cc = CircuitComponent.from_bargmann(coh.bargmann, modes_out_ket=[0])
            >>> assert isinstance(coh_cc, CircuitComponent)
            >>> assert coh == coh_cc  # equality looks at representation and wires
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
        r"""Returns a circuit component from the given triple (A,b,c) that parametrizes the
        quadrature wavefunction of this component in the form :math:`c * exp(1/2 x^T A x + b^T x)`.

        Args:
            modes_out_bra: The output modes on the bra side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_ket: The input modes on the ket side of this component.
            triple: The (A,b,c) triple that parametrizes the wave function.
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature, ``phi=pi/2`` to the p quadrature. The default value is ``0``.
            name: The name of this component.

        Returns:
            A circuit component with the given quadrature representation.
        """
        from mrmustard.lab_dev.circuit_components_utils import BtoQ

        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        QtoB_ob = BtoQ(modes_out_bra, phi).inverse().adjoint  # output bra
        QtoB_ib = BtoQ(modes_in_bra, phi).inverse().adjoint.dual  # input bra
        QtoB_ok = BtoQ(modes_out_ket, phi).inverse()  # output ket
        QtoB_ik = BtoQ(modes_in_ket, phi).inverse().dual  # input ket
        # NOTE: the representation is Bargmann here because we use the inverse of BtoQ on the B side
        QQQQ = CircuitComponent._from_attributes(Bargmann(*triple), wires)
        BBBB = QtoB_ib @ (QtoB_ik @ QQQQ @ QtoB_ok) @ QtoB_ob
        return cls._from_attributes(BBBB.representation, wires, name)

    def quadrature(self, phi: float = 0.0) -> tuple | ComplexTensor:
        r"""
        The quadrature representation data of this circuit component.
        """
        from mrmustard.lab_dev.circuit_components_utils import BtoQ

        BtoQ_ob = BtoQ(self.wires.output.bra.modes, phi).adjoint
        BtoQ_ib = BtoQ(self.wires.input.bra.modes, phi).adjoint.dual
        BtoQ_ok = BtoQ(self.wires.output.ket.modes, phi)
        BtoQ_ik = BtoQ(self.wires.input.ket.modes, phi).dual
        QQQQ = BtoQ_ib @ (BtoQ_ik @ self @ BtoQ_ok) @ BtoQ_ob
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
        The number of modes spanned by this component across all wires.
        """
        return len(self.modes)

    @property
    def name(self) -> str:
        r"""
        The name of this component.
        """
        if self._name is None:
            name = self.short_name
            modes = "".join(str(m) for m in sorted(self.wires.modes))
            self._name = name + modes if len(modes) < 5 else name
        return self._name

    @property
    def parameter_set(self) -> ParameterSet:
        r"""
        The set of parameters of this component.
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
        The adjoint of this component obtained by conjugating the representation and swapping
        the ket and bra wires. The returned object is a view of the original component which
        applies a conjugation and a swap of the wires, but does not copy the data in memory.
        """
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""
        The dual of this component obtained by conjugating the representation and swapping
        the input and output wires. The returned object is a view of the original component which
        applies a conjugation and a swap of the wires, but does not copy the data in memory.
        """
        return DualView(self)

    def _light_copy(self, wires: Optional[Wires] = None) -> CircuitComponent:
        r"""
        Creates a "light" copy of this component by referencing its __dict__, except for the wires,
        which are a new object or the given one.
        This is useful when one needs the same component acting on different modes, for example.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.__dict__["_wires"] = wires or Wires(*self.wires.args)
        return instance

    def on(self, modes: Sequence[int]) -> CircuitComponent:
        r"""
        Creates a light copy of this component that acts on the given ``modes`` instead of the
        original modes. It only works if the component's wires are all defined on the same modes.
        As a light copy, the returned component shares the representation with the original one.

        If a more general rewiring is needed, while maintaining a light copy to the original, use
        ``._light_copy(new_wires)`` and pass the desired wires.

        Args:
            modes: The new modes that this component acts on.

        Returns:
            The component acting on the specified modes.

        Raises:
            ValueError: If the component's wires are not all defined on the same modes or if the
            length of the given modes is different from the length of the original modes.
        """
        ob = self.wires.output.bra.modes
        ib = self.wires.input.bra.modes
        ok = self.wires.output.ket.modes
        ik = self.wires.input.ket.modes
        subsets = [s for s in (ob, ib, ok, ik) if s]
        if any(s != subsets[0] for s in subsets):
            raise ValueError(
                f"Cannot rewire a component with wires on different modes ({ob, ib, ok, ik})."
            )
        for subset in subsets:
            if subset and len(subset) != len(modes):
                raise ValueError(f"Expected ``{len(modes)}`` modes, found ``{len(subset)}``.")
        ret = self._light_copy()
        modes = set(modes)
        ret._wires = Wires(
            modes_out_bra=modes if ob else set(),
            modes_in_bra=modes if ib else set(),
            modes_out_ket=modes if ok else set(),
            modes_in_ket=modes if ik else set(),
        )

        return ret

    def to_fock(self, shape: Optional[Union[int, Iterable[int]]] = None) -> CircuitComponent:
        r"""
        Returns a new circuit component with the same attributes as this and a ``Fock`` representation.

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
        Implements the multiplication by a scalar from the right.
        """
        return self._from_attributes(self.representation * other, self.wires, self.name)

    def __rmul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar from the left.
        """
        return self * other

    def __truediv__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the division by a scalar for circuit components.
        """
        return self._from_attributes(self.representation / other, self.wires, self.name)

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations and wires, but not the other attributes (e.g. name and parameter set).
        """
        return self.representation == other.representation and self.wires == other.wires

    def _matmul_indices(self, other: CircuitComponent) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""
        Finds the indices of the wires being contracted when ``self @ other`` is called.
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

    def __matmul__(self, other: CircuitComponent | Scalar) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` without adding adjoints.
        It allows for a more custom way of contracting components.
        """
        if isinstance(other, (numbers.Number, np.ndarray)):
            return self * other
        wires_result, perm = self.wires @ other.wires
        idx_z, idx_zconj = self._matmul_indices(other)
        rep = self.representation[idx_z] @ other.representation[idx_zconj]
        rep = rep.reorder(perm) if perm else rep
        return CircuitComponent._from_attributes(rep, wires_result, None)

    def __rmatmul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Multiplies a scalar with a circuit component when written as ``scalar @ component``.
        """
        return self * other

    def __rshift__(self, other: CircuitComponent | numbers.Number) -> CircuitComponent | np.ndarray:
        r"""
        Contracts ``self`` and ``other`` (output of self going into input of other).
        It adds the adjoints when they are missing (e.g. if ``self`` is a Ket and
        ``other`` is a Channel). An error is raised if these cannot be deduced from
        the wires of the components. For example this allows ``Ket``s to be right-shifted
        into ``Channel``s and automatically the result is a ``DM``. If the result has
        no wires left, it returns the (batched) scalar value of the representation.
        Note that a ``CircuitComponent`` is allowed to right-shift into scalars because the scalar
        part may result from an automated contraction subroutine that involves several components).

        .. code-block::
            >>> from mrmustard.lab_dev import Coherent, Attenuator, Ket, DM, Channel
            >>> import numpy as np
            >>> assert issubclass(Coherent, Ket)
            >>> assert issubclass(Attenuator, Channel)
            >>> assert isinstance(Coherent([0], 1.0) >> Attenuator([0], 0.5), DM)
            >>> assert isinstance(Coherent([0], 1.0) >> Coherent([0], 1.0).dual, complex)
        """
        if hasattr(other, "__custom_rrshift__"):
            return other.__custom_rrshift__(self)

        if isinstance(other, (numbers.Number, np.ndarray)):
            return self * other

        msg = f"``>>`` not supported between {self} and {other} because it's not clear "
        msg += "whether or where to add missing wires. Use ``@`` and specify all the components."

        only_ket = not self.wires.bra and not other.wires.bra
        only_bra = not self.wires.ket and not other.wires.ket
        both_sides = self.wires.bra and self.wires.ket and other.wires.bra and other.wires.ket
        if only_ket or only_bra or both_sides:
            return self._rshift_return(self @ other)

        self_needs_bra = (not self.wires.bra) and other.wires.bra and other.wires.ket
        self_needs_ket = (not self.wires.ket) and other.wires.bra and other.wires.ket
        if self_needs_bra or self_needs_ket:
            return self._rshift_return(self.adjoint @ (self @ other))

        other_needs_bra = (self.wires.bra and self.wires.ket) and not other.wires.bra
        other_needs_ket = (self.wires.bra and self.wires.ket) and not other.wires.ket
        if other_needs_bra or other_needs_ket:
            return self._rshift_return((self @ other) @ other.adjoint)

        raise ValueError(msg)

    def _rshift_return(
        self, ret: CircuitComponent | np.ndarray | complex
    ) -> CircuitComponent | np.ndarray | complex:
        "internal convenience method for right-shift, to return the right type of object"
        if len(ret.wires) > 0:
            return ret
        scalar = ret.representation.scalar
        return math.sum(scalar) if not settings.UNSAFE_ZIP_BATCH else scalar

    def __rrshift__(self, other: Scalar) -> CircuitComponent | np.array:
        r"""
        Multiplies a scalar with a circuit component when written as ``scalar >> component``.
        This is needed when the "component" on the left is the result of a contraction that leaves
        no wires and the component is returned as a scalar. Note that there is an edge case if the object on the left happens to have the ``__rshift__`` method, but it's not the one we want (usually `>>` is about bit shifts) like a numpy array. In this case in an expression with types ``np.ndarray >> CircuitComponent`` the method ``__rrshift__`` will not be called, and something else will be returned.
        """
        ret = self * other
        try:
            return ret.representation.scalar
        except AttributeError:
            return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modes={self.modes}, name={self.name or None})"

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
    r"""A base class for views of circuit components. It allows for a more efficient
    use of components when we need the same component on different wires.

    Args:
        component: The circuit component to take the view of.
    """

    def __init__(self, component: CircuitComponent) -> None:
        self.__dict__ = component.__dict__.copy()
        self._component = component._light_copy()

    def __getattr__(self, name):
        r"""send calls to the component"""
        return getattr(self._component, name)

    def __repr__(self) -> str:
        return repr(self._component)


class AdjointView(CCView):
    r"""
    Adjoint view of a circuit component obtained by swapping the ket/bra wires
    and conjugating the representation. Note the representation is a wrapper
    property around the original one, so it can work also for classes whose
    representation attribute is a computed property like the trainable components.

    Args:
        component: The circuit component to take the view of.
    """

    @property
    def short_name(self) -> str:
        "short name that appears in the circuit"
        return self._component.short_name + "_adj"

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        Returns a light-copy of the component that was used to generate the view.
        """
        return self._component._light_copy()

    @property
    def representation(self):
        r"""
        A representation of this circuit component. Note that ket and bra indices
        have been swapped.
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
    Dual view of a circuit component obtained by swapping the input/output wires
    and conjugating the representation. Note the representation is a wrapper
    property around the original one, so it can work also for classes whose
    representation attribute is a computed property like the trainable components.

    Args:
        component: The circuit component to take the view of.
    """

    @property
    def short_name(self) -> str:
        "short name that appears in the circuit"
        return self._component.short_name + "_dual"

    @property
    def dual(self) -> CircuitComponent:
        r"""
        Returns a light-copy of the component that was used to generate the view.
        """
        return self._component._light_copy()

    @property
    def representation(self):
        r"""
        A representation of this circuit component. Note that input and output indices
        have been swapped.
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
