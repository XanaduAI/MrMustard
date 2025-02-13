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

# pylint: disable=super-init-not-called, import-outside-toplevel
from __future__ import annotations

from inspect import signature
from pydoc import locate
from typing import Any, Sequence
import numbers
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike
import ipywidgets as widgets
from IPython.display import display

from mrmustard import settings, math, widgets as mmwidgets
from mrmustard.utils.typing import (
    Scalar,
    ComplexTensor,
    ComplexMatrix,
    ComplexVector,
    Vector,
    Batch,
)
from mrmustard.physics.ansatz import Ansatz, PolyExpAnsatz, ArrayAnsatz
from mrmustard.physics.fock_utils import quadrature_basis
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.physics.wires import Wires
from mrmustard.physics.representations import Representation

__all__ = ["CircuitComponent"]


class CircuitComponent:
    r"""
    A base class for the circuit components (states, transformations, measurements,
    and any component made by combining CircuitComponents). CircuitComponents are
    defined by their ``representation``. See :class:`Representation` for more details.

    Args:
        representation: The representation of this circuit component.
        name: The name of this component.
    """

    short_name = "CC"

    def __init__(
        self,
        representation: Representation | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name
        self._parameters = ParameterSet()
        self._representation = representation or Representation()

    def _serialize(self) -> tuple[dict[str, Any], dict[str, ArrayLike]]:
        """
        Inner serialization to be used by Circuit.serialize().

        The first dict must be JSON-serializable, and the second dict must contain
        the (non-JSON-serializable) array-like data to be collected separately.
        """
        cls = type(self)
        serializable = {"class": f"{cls.__module__}.{cls.__qualname__}"}
        params = signature(cls).parameters
        if "name" in params:  # assume abstract type, serialize the representation
            ansatz_cls = type(self.ansatz)
            serializable["name"] = self.name
            serializable["wires"] = tuple(tuple(a) for a in self.wires.args)
            serializable["ansatz_cls"] = f"{ansatz_cls.__module__}.{ansatz_cls.__qualname__}"
            return serializable, self.ansatz.to_dict()

        # handle modes parameter
        if "modes" in params:
            serializable["modes"] = tuple(self.wires.modes)
        elif "modes_out" in params and "modes_in" in params:
            serializable["modes_out"] = tuple(self.wires.output.modes)
            serializable["modes_in"] = tuple(self.wires.input.modes)
        else:
            raise TypeError(f"{cls.__name__} does not seem to have any wires construction method")

        if self.parameters:
            for k, v in self.parameters.variables.items():
                serializable[f"{k}_bounds"] = v.bounds
                serializable[f"{k}_trainable"] = True
            return serializable, {k: v.value for k, v in self.parameters.all_parameters.items()}

        return serializable, {}

    @classmethod
    def _deserialize(cls, data: dict) -> CircuitComponent:
        r"""
        Deserialization when within a circuit.
        """
        if "ansatz_cls" in data:
            ansatz_cls, wires, name = map(data.pop, ["ansatz_cls", "wires", "name"])
            ansatz = locate(ansatz_cls).from_dict(data)
            return cls._from_attributes(
                Representation(ansatz, Wires(*tuple(set(m) for m in wires))), name=name
            )

        return cls(**data)

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        The adjoint of this component obtained by conjugating the representation and swapping
        the ket and bra wires.
        """
        ret = CircuitComponent(self.representation.adjoint, self.name)
        ret.short_name = self.short_name
        for param in self.parameters.all_parameters.values():
            ret.parameters.add_parameter(param)
        return ret

    @property
    def dual(self) -> CircuitComponent:
        r"""
        The dual of this component obtained by conjugating the representation and swapping
        the input and output wires.
        """
        ret = CircuitComponent(self.representation.dual, self.name)
        ret.short_name = self.short_name
        for param in self.parameters.all_parameters.values():
            ret.parameters.add_parameter(param)
        return ret

    @cached_property
    def manual_shape(self) -> list[int | None]:
        r"""
        The shape of this Component in the Fock representation. If not manually set,
        it is a list of M ``None``s where M is the number of wires of the component.
        The manual_shape is a list and therefore it is mutable. In fact, it can evolve
        over time as we learn more about the component or its neighbours. For
        each wire, the entry is either an integer or ``None``. If it is an integer, it
        is the dimension of the corresponding Fock space. If it is ``None``, it means
        the best shape is not known yet. ``None``s automatically become integers when
        ``auto_shape`` is called, but the integers already set are not changed.
        The order of the elements in the shape is intended the same order as the wires
        in the `.wires` attribute.
        """
        try:  # to read it from array ansatz
            return list(self.ansatz.array.shape[1:])
        except AttributeError:  # bargmann
            return [None] * len(self.wires)

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
        if self._name is None:
            name = self.short_name
            modes = "".join(str(m) for m in sorted(self.wires.modes))
            self._name = name + modes if len(modes) < 5 else name
        return self._name

    @property
    def n_modes(self) -> int:
        r"""
        The number of modes spanned by this component across all wires.
        """
        return len(self.modes)

    @property
    def parameters(self) -> ParameterSet:
        r"""
        The set of parameters of this component.
        """
        return self._parameters

    @property
    def ansatz(self) -> Ansatz | None:
        r"""
        The ansatz of this circuit component.
        """
        return self._representation.ansatz

    @property
    def representation(self) -> Representation | None:
        r"""
        The representation of this circuit component.
        """
        return self._representation

    @property
    def wires(self) -> Wires:
        r"""
        The wires of this component.
        """
        return self._representation.wires

    @classmethod
    def from_bargmann(
        cls,
        triple: tuple,
        modes_out_bra: Sequence[int] = (),
        modes_in_bra: Sequence[int] = (),
        modes_out_ket: Sequence[int] = (),
        modes_in_ket: Sequence[int] = (),
        name: str | None = None,
    ) -> CircuitComponent:  # pylint:disable=too-many-positional-arguments
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
        ansatz = PolyExpAnsatz(*triple)
        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        return cls._from_attributes(Representation(ansatz, wires), name)

    @classmethod
    def from_quadrature(
        cls,
        modes_out_bra: Sequence[int],
        modes_in_bra: Sequence[int],
        modes_out_ket: Sequence[int],
        modes_in_ket: Sequence[int],
        triple: tuple,
        phi: float = 0.0,
        name: str | None = None,
    ) -> CircuitComponent:  # pylint:disable=too-many-positional-arguments
        r"""
        Returns a circuit component from the given triple (A,b,c) that parametrizes the
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
        from .circuit_components_utils.b_to_q import BtoQ

        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        QtoB_ob = BtoQ(modes_out_bra, phi).inverse().adjoint  # output bra
        QtoB_ib = BtoQ(modes_in_bra, phi).inverse().adjoint.dual  # input bra
        QtoB_ok = BtoQ(modes_out_ket, phi).inverse()  # output ket
        QtoB_ik = BtoQ(modes_in_ket, phi).inverse().dual  # input ket
        # NOTE: the representation is Bargmann here because we use the inverse of BtoQ on the B side
        QQQQ = CircuitComponent(Representation(PolyExpAnsatz(*triple), wires))
        BBBB = QtoB_ib @ (QtoB_ik @ QQQQ @ QtoB_ok) @ QtoB_ob
        return cls._from_attributes(Representation(BBBB.ansatz, wires), name)

    def to_quadrature(self, phi: float = 0.0) -> CircuitComponent:
        r"""
        Returns a circuit component with the quadrature representation of this component
        in terms of A,b,c.

        Args:
            phi (float): The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            A circuit component with the given quadrature representation.
        """
        from .circuit_components_utils.b_to_q import BtoQ

        BtoQ_ob = BtoQ(self.wires.output.bra.modes, phi).adjoint
        BtoQ_ib = BtoQ(self.wires.input.bra.modes, phi).adjoint.dual
        BtoQ_ok = BtoQ(self.wires.output.ket.modes, phi)
        BtoQ_ik = BtoQ(self.wires.input.ket.modes, phi).dual

        object_to_convert = self
        if isinstance(self.ansatz, ArrayAnsatz):
            object_to_convert = self.to_bargmann()

        QQQQ = BtoQ_ib @ (BtoQ_ik @ object_to_convert @ BtoQ_ok) @ BtoQ_ob
        return QQQQ

    def quadrature_triple(
        self, phi: float = 0.0
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The quadrature representation triple A,b,c of this circuit component.

        Args:
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            A,b,c triple of the quadrature representation
        """
        return self.to_quadrature(phi=phi).ansatz.data

    def quadrature(self, quad: Batch[Vector], phi: float = 0.0) -> ComplexTensor:
        r"""
        The (discretized) quadrature basis representation of the circuit component.
        Args:
            quad: discretized quadrature points to evaluate over in the
                quadrature representation
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            A circuit component with the given quadrature representation.
        """

        if isinstance(self.ansatz, ArrayAnsatz):
            fock_arrays = self.ansatz.array
            # Find where all the bras and kets are so they can be conjugated appropriately
            conjugates = [i not in self.wires.ket.indices for i in range(len(self.wires.indices))]
            quad_basis = math.sum(
                math.astensor([quadrature_basis(array, quad, conjugates, phi) for array in fock_arrays]), axis=0
            )
            return quad_basis

        QQQQ = self.to_quadrature(phi=phi)
        return QQQQ.ansatz(quad)

    @classmethod
    def _from_attributes(
        cls,
        representation: Representation,
        name: str | None = None,
    ) -> CircuitComponent:
        r"""
        Initializes a circuit component from a ``Representation`` and a name.
        It differs from the __init__ in that the return type is the closest parent
        among the types ``Ket``, ``DM``, ``Unitary``, ``Operation``, ``Channel``,
        and ``Map``. This is to ensure the right properties are used when calling
        methods on the returned object, e.g. when adding two coherent states we
        don't get a generic ``CircuitComponent`` but a ``Ket``:

        .. code-block::
            >>> from mrmustard.lab_dev import Coherent, Ket
            >>> cat = Coherent(modes=[0], x=2.0) + Coherent(modes=[0], x=-2.0)
            >>> assert isinstance(cat, Ket)

        Args:
            representation: A representation for this circuit component.
            name: The name for this component (optional).

        Returns:
            A circuit component with the given attributes.
        """
        types = {"Ket", "DM", "Unitary", "Operation", "Channel", "Map"}
        for tp in cls.mro():
            if tp.__name__ in types:
                return tp(representation=representation, name=name)
        return CircuitComponent(representation, name)

    def auto_shape(self, **_) -> tuple[int, ...]:
        r"""
        The shape of the Fock representation of this component. If the component has a Fock representation
        then it is just the shape of the array. If the components is a State in Bargmann
        representation the shape is calculated using autoshape using the single-mode marginals.
        If the component is not a State then the shape is a tuple of ``settings.AUTOSHAPE_MAX`` values
        except where the ``manual_shape`` attribute has been set..
        """
        return tuple(s or settings.AUTOSHAPE_MAX for s in self.manual_shape)

    def bargmann_triple(
        self, batched: bool = False
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The Bargmann parametrization of this component, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this component is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`

        If ``batched`` is ``False`` (default), it removes the batch dimension if it is of size 1.

        .. code-block:: pycon

            >>> from mrmustard.lab_dev import CircuitComponent, Coherent
            >>> coh = Coherent(modes=[0], x=1.0)
            >>> coh_cc = CircuitComponent.from_bargmann(coh.bargmann_triple(), modes_out_ket=[0])
            >>> assert isinstance(coh_cc, CircuitComponent)
            >>> assert coh == coh_cc  # equality looks at representation and wires
        """
        return self._representation.bargmann_triple(batched)

    def fock_array(self, shape: int | Sequence[int] | None = None, batched=False) -> ComplexTensor:
        r"""
        Returns an array representation of this component in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is estimated.
            batched: Whether the returned representation is batched or not. If ``False`` (default)
                it will squeeze the batch dimension if it is 1.
        Returns:
            array: The Fock representation of this component.
        """
        return self._representation.fock_array(shape or self.auto_shape(), batched)

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
        ret = self._light_copy(
            Wires(
                modes_out_bra=set(modes) if ob else set(),
                modes_in_bra=set(modes) if ib else set(),
                modes_out_ket=set(modes) if ok else set(),
                modes_in_ket=set(modes) if ik else set(),
            )
        )
        return ret

    def to_bargmann(self) -> CircuitComponent:
        r"""
        Returns a new circuit component with the same attributes as this and a ``Bargmann`` representation.
        .. code-block::

            >>> from mrmustard.lab_dev import Dgate
            >>> from mrmustard.physics.ansatz import PolyExpAnsatz

            >>> d = Dgate([1], x=0.1, y=0.1)
            >>> d_fock = d.to_fock(shape=3)
            >>> d_bargmann = d_fock.to_bargmann()

            >>> assert d_bargmann.name == d.name
            >>> assert d_bargmann.wires == d.wires
            >>> assert isinstance(d_bargmann.ansatz, PolyExpAnsatz)
        """
        rep = self._representation.to_bargmann()
        try:
            ret = self._getitem_builtin(self.modes)
            ret._representation = rep
        except TypeError:
            ret = self._from_attributes(rep, self.name)
        if "manual_shape" in ret.__dict__:
            del ret.manual_shape
        return ret

    def to_fock(self, shape: int | Sequence[int] | None = None) -> CircuitComponent:
        r"""
        Returns a new circuit component with the same attributes as this and a ``Fock`` representation.

        .. code-block::

            >>> from mrmustard.lab_dev import Dgate
            >>> from mrmustard.physics.ansatz import ArrayAnsatz

            >>> d = Dgate([1], x=0.1, y=0.1)
            >>> d_fock = d.to_fock(shape=3)

            >>> assert d_fock.name == d.name
            >>> assert isinstance(d_fock.ansatz, ArrayAnsatz)

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        """
        rep = self._representation.to_fock(shape or self.auto_shape())
        try:
            ret = self._getitem_builtin(self.modes)
            ret._representation = rep
        except TypeError:
            ret = self._from_attributes(rep, self.name)
        if "manual_shape" in ret.__dict__:
            del ret.manual_shape
        return ret

    def _getitem_builtin(self, modes: set[int]):
        r"""
        A convenience function to slice built-in circuit components (CCs).

        Built-in CCs come with a parameter set. To slice them, we simply slice the parameter
        set, and then used the sliced parameter set to re-initialize them.

        This approach avoids computing the representation, which may be expensive. Additionally,
        it allows returning trainable CCs.
        """
        items = [i for i, m in enumerate(self.modes) if m in modes]
        kwargs = self.parameters[items].to_dict()
        return self.__class__(modes=modes, **kwargs)

    def _light_copy(self, wires: Wires | None = None) -> CircuitComponent:
        r"""
        Creates a "light" copy of this component by referencing its __dict__, except for the wires,
        which are a new object or the given one.
        This is useful when one needs the same component acting on different modes, for example.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance.__dict__["_representation"] = Representation(
            self.ansatz, wires or Wires(*self.wires.args)
        )
        return instance

    def _rshift_return(
        self, result: CircuitComponent | np.ndarray | complex
    ) -> CircuitComponent | np.ndarray | complex:
        "internal convenience method for right-shift, to return the right type of object"
        if len(result.wires) > 0:
            return result
        scalar = result.ansatz.scalar
        return math.sum(scalar) if not settings.UNSAFE_ZIP_BATCH else scalar

    def __add__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Implements the addition between circuit components.
        """
        if self.wires != other.wires:
            raise ValueError("Cannot add components with different wires.")
        ansatz = self.ansatz + other.ansatz
        name = self.name if self.name == other.name else ""
        return self._from_attributes(Representation(ansatz, self.wires), name)

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations, but not the other attributes
        (e.g. name and parameter set).
        """
        if isinstance(other, CircuitComponent):
            return self._representation == other._representation
        return False

    def __matmul__(self, other: CircuitComponent | Scalar) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` without adding adjoints.
        It allows for contracting components exactly as specified.

        For example, a coherent state can be input to an attenuator, but
        the attenuator has two inputs: on the ket and the bra side.
        The ``>>`` operator would automatically add the adjoint of the coherent
        state on the bra side of the input of the attenuator, but the ``@`` operator
        instead does not:

        .. code-block::
            >>> from mrmustard.lab_dev import Coherent, Attenuator
            >>> coh = Coherent([0], 1.0)
            >>> att = Attenuator([0], 0.5)
            >>> assert (coh @ att).wires.input.bra  # the input bra is still uncontracted
        """
        if isinstance(other, (numbers.Number, np.ndarray)):
            return self * other
        result = self._representation @ other._representation
        return CircuitComponent(result, None)

    def __mul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar from the right.
        """
        return self._from_attributes(Representation(self.ansatz * other, self.wires), self.name)

    def __repr__(self) -> str:
        ansatz = self.ansatz
        repr_name = ansatz.__class__.__name__
        if repr_name == "NoneType":
            return self.__class__.__name__ + f"(modes={self.modes}, name={self.name})"
        else:
            return (
                self.__class__.__name__
                + f"(modes={self.modes}, name={self.name}"
                + f", repr={repr_name})"
            )

    def __rmatmul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Multiplies a scalar with a circuit component when written as ``scalar @ component``.
        """
        return self * other

    def __rmul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar from the left.
        """
        return self * other

    def __rrshift__(self, other: Scalar) -> CircuitComponent | np.array:
        r"""
        Multiplies a scalar with a circuit component when written as ``scalar >> component``.
        This is needed when the "component" on the left is the result of a contraction that leaves
        no wires and the component is returned as a scalar. Note that there is an edge case if the
        object on the left happens to have the ``__rshift__`` method, but it's not the one we want
        (usually `>>` is about bit shifts) like a numpy array. In this case in an expression with
        types ``np.ndarray >> CircuitComponent`` the method ``CircuitComponent.__rrshift__`` will
        not be called, and something else will be returned.
        """
        ret = self * other
        return ret.ansatz.scalar

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

        s_k = self.wires.ket
        s_b = self.wires.bra
        o_k = other.wires.ket
        o_b = other.wires.bra

        only_ket = (not s_b and s_k) and (not o_b and o_k)
        only_bra = (not s_k and s_b) and (not o_k and o_b)
        both_sides = s_b and s_k and o_b and o_k

        self_needs_bra = (not s_b and s_k) and (o_b and o_k)
        self_needs_ket = (not s_k and s_b) and (o_b and o_k)

        other_needs_bra = (s_b and s_k) and (not o_b and o_k)
        other_needs_ket = (s_b and s_k) and (not o_k and o_b)

        if only_ket or only_bra or both_sides:
            ret = self @ other
        elif self_needs_bra or self_needs_ket:
            ret = self.adjoint @ (self @ other)
        elif other_needs_bra or other_needs_ket:
            ret = (self @ other.adjoint) @ other
        else:
            msg = f"``>>`` not supported between {self} and {other} because it's not clear "
            msg += "whether or where to add bra wires. Use ``@`` instead and specify all the components."
            raise ValueError(msg)
        return self._rshift_return(ret)

    def __sub__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Implements the subtraction between circuit components.
        """
        if self.wires != other.wires:
            raise ValueError("Cannot subtract components with different wires.")
        ansatz = self.ansatz - other.ansatz
        name = self.name if self.name == other.name else ""
        return self._from_attributes(Representation(ansatz, self.wires), name)

    def __truediv__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the division by a scalar for circuit components.
        """
        return self._from_attributes(Representation(self.ansatz / other, self.wires), self.name)

    def _ipython_display_(self):
        if mmwidgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        # both reps might return None
        rep_fn = mmwidgets.fock if isinstance(self.ansatz, ArrayAnsatz) else mmwidgets.bargmann
        rep_widget = rep_fn(self.ansatz)
        wires_widget = mmwidgets.wires(self.wires)
        if not rep_widget:
            title_widget = widgets.HTML(f"<h1>{self.name or type(self).__name__}</h1>")
            display(widgets.VBox([title_widget, wires_widget]))
            return
        rep_widget.layout.padding = "10px"
        wires_widget.layout.padding = "10px"
        display(widgets.Box([wires_widget, rep_widget]))
