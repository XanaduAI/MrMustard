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

import numbers
from collections.abc import Sequence
from inspect import signature
from pydoc import locate
from typing import Any

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from numpy.typing import ArrayLike

from mrmustard import math, settings
from mrmustard import widgets as mmwidgets
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Variable
from mrmustard.physics.ansatz import Ansatz, ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.fock_utils import oscillator_eigenstate
from mrmustard.physics.triples import identity_Abc
from mrmustard.physics.utils import outer_product_batch_str, zip_batch_strings
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    RealVector,
    Scalar,
)

__all__ = ["CircuitComponent"]


class CircuitComponent:
    r"""
    A base class for the circuit components (states, transformations, measurements,
    and any component made by combining CircuitComponents). CircuitComponents are
    defined by their ``ansatz`` and ``wires``.

    Args:
        ansatz: The ansatz of this circuit component.
        wires: The wires of this circuit component.
        name: The name of this circuit component.
    """

    short_name = "CC"

    def __init__(
        self,
        ansatz: Ansatz | None = None,
        wires: Wires | None = None,
        name: str | None = None,
    ) -> None:
        self._ansatz = ansatz
        self._name = name
        self._parameters = ParameterSet()
        self._wires = wires or Wires(set(), set(), set(), set())

        if isinstance(ansatz, ArrayAnsatz):
            for w in self.wires.quantum:
                w.repr = ReprEnum.FOCK
                w.fock_cutoff = ansatz.core_shape[w.index]

    @property
    def adjoint(self) -> CircuitComponent:
        r"""
        The adjoint of this component obtained by conjugating the ansatz and swapping
        the ket and bra wires.

        .. code-block::

            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0])
            >>> assert psi.dm() == psi.contract(psi.adjoint)
        """
        bras = self.wires.bra.indices
        kets = self.wires.ket.indices
        ansatz = self.ansatz.reorder(kets + bras).conj if self.ansatz else None
        ret = CircuitComponent(ansatz, self.wires.adjoint, name=self.name)
        ret.short_name = self.short_name
        for param in self.parameters.all_parameters.values():
            ret.parameters.add_parameter(param)
        return ret

    @property
    def ansatz(self) -> Ansatz:
        r"""
        The ansatz of this circuit component.
        """
        return self._ansatz

    @property
    def dual(self) -> CircuitComponent:
        r"""
        The dual of this component obtained by conjugating the ansatz and swapping
        the input and output wires.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0])
            >>> assert math.allclose(1.0, psi >> psi.dual)
        """
        ok = self.wires.ket.output.indices
        ik = self.wires.ket.input.indices
        ib = self.wires.bra.input.indices
        ob = self.wires.bra.output.indices
        ansatz = self.ansatz.reorder(ib + ob + ik + ok).conj if self.ansatz else None
        ret = CircuitComponent(ansatz, self.wires.dual, name=self.name)
        ret.short_name = self.short_name
        for param in self.parameters.all_parameters.values():
            ret.parameters.add_parameter(param)
        return ret

    @property
    def manual_shape(self) -> tuple[int | None]:
        r"""
        The shape of this Component in the Fock representation. If not manually set,
        it is a tuple of M ``None``s where M is the number of wires of the component. For
        each wire, the entry is either an integer or ``None``. If it is an integer, it
        is the cutoff of the corresponding Fock space. If it is ``None``, it means
        the best shape is not known yet. ``None``s automatically become integers when
        ``auto_shape`` is called, but the integers already set are not changed.
        The order of the elements in the shape is intended the same order as the wires
        in the `.sorted_wires` attribute.
        """
        return tuple(w.fock_cutoff for w in self.wires.quantum.sorted_wires)

    @manual_shape.setter
    def manual_shape(self, shape: tuple[int | None]):
        for w, s in zip(self.wires.quantum.sorted_wires, shape):
            w.fock_cutoff = s

    @property
    def modes(self) -> list[int]:
        r"""
        The sorted list of modes of this component.

        .. code-block::

            >>> from mrmustard.lab import Ket

            >>> ket = Ket.random([0, 1])
            >>> assert ket.modes == (0, 1)
        """
        return tuple(sorted(self.wires.modes))

    @property
    def name(self) -> str:
        r"""
        The name of this component.

        .. code-block::

            >>> from mrmustard.lab import BtoPS

            >>> assert BtoPS(modes=0, s=0).name == "BtoPS"
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

        .. code-block::

            >>> from mrmustard.lab import Ket

            >>> ket = Ket.random([0, 1])
            >>> assert ket.n_modes == 2
        """
        return len(self.modes)

    @property
    def parameters(self) -> ParameterSet:
        r"""
        The set of parameters of this component.

        .. code-block::

            >>> from mrmustard.lab import Coherent

            >>> coh = Coherent(mode=0, x=1.0)
            >>> assert coh.parameters.x.value == 1.0
        """
        return self._parameters

    @property
    def wires(self) -> Wires:
        r"""
        The wires of this circuit component.
        """
        return self._wires

    @classmethod
    def from_bargmann(
        cls,
        triple: tuple,
        modes_out_bra: Sequence[int] = (),
        modes_in_bra: Sequence[int] = (),
        modes_out_ket: Sequence[int] = (),
        modes_in_ket: Sequence[int] = (),
        name: str | None = None,
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

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import CircuitComponent, Identity
            >>> from mrmustard.physics.ansatz import PolyExpAnsatz

            >>> A = math.astensor([[0, 1], [1, 0]])
            >>> b = math.astensor([0, 0])
            >>> c = 1
            >>> modes_out_bra = {}
            >>> modes_in_bra = {}
            >>> modes_out_ket = {0}
            >>> modes_in_ket = {0}
            >>> triple = (A, b, c)
            >>> cc = CircuitComponent.from_bargmann(triple, modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)

            >>> assert isinstance(cc.ansatz, PolyExpAnsatz)
            >>> assert cc == Identity(modes = 0)
        """
        ansatz = PolyExpAnsatz(*triple)
        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        return cls._from_attributes(ansatz, wires, name)

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
    ) -> CircuitComponent:
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
        from .circuit_components_utils.b_to_q import BtoQ  # noqa: PLC0415

        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        QtoB_ob = BtoQ(modes_out_bra, phi).inverse().adjoint  # output bra
        QtoB_ib = BtoQ(modes_in_bra, phi).inverse().adjoint.dual  # input bra
        QtoB_ok = BtoQ(modes_out_ket, phi).inverse()  # output ket
        QtoB_ik = BtoQ(modes_in_ket, phi).inverse().dual  # input ket
        # NOTE: the representation is Bargmann here because we use the inverse of BtoQ on the B side
        QQQQ = CircuitComponent(PolyExpAnsatz(*triple), wires)
        BBBB = QtoB_ib.contract(QtoB_ik.contract(QQQQ).contract(QtoB_ok)).contract(QtoB_ob)
        return cls._from_attributes(BBBB.ansatz, wires, name)

    @classmethod
    def _deserialize(cls, data: dict) -> CircuitComponent:
        r"""
        Deserialization when within a circuit.

        Args:
            data: The data to deserialize.

        Returns:
            A circuit component with the given serialized data.
        """
        if "ansatz_cls" in data:
            ansatz_cls, wires, name = map(data.pop, ["ansatz_cls", "wires", "name"])
            ansatz = locate(ansatz_cls).from_dict(data)
            return cls._from_attributes(ansatz, Wires(*tuple(set(m) for m in wires)), name=name)
        if "modes" in data:
            data["modes"] = tuple(data["modes"])
        elif "mode" in data:
            data["mode"] = tuple(data["mode"])
        return cls(**data)

    @classmethod
    def _from_attributes(
        cls,
        ansatz: Ansatz,
        wires: Wires,
        name: str | None = None,
    ) -> CircuitComponent:
        r"""
        Initializes a circuit component from an ``Ansatz``, ``Wires`` and a name.
        It differs from the __init__ in that the return type is the closest parent
        among the types ``Ket``, ``DM``, ``Unitary``, ``Operation``, ``Channel``,
        and ``Map``. This is to ensure the right properties are used when calling
        methods on the returned object, e.g. when adding two coherent states we
        don't get a generic ``CircuitComponent`` but a ``Ket``:

        .. code-block::

            >>> from mrmustard.lab import Coherent, Ket
            >>> cat = Coherent(mode=0, x=2.0) + Coherent(mode=0, x=-2.0)
            >>> assert isinstance(cat, Ket)

        Args:
            ansatz: An ansatz for this circuit component.
            wires: The wires for this circuit component.
            name: The name for this component (optional).

        Returns:
            A circuit component with the given attributes.
        """
        types = {"Ket", "DM", "Unitary", "Operation", "Channel", "Map"}
        for tp in cls.mro():
            if tp.__name__ in types:
                return tp(ansatz=ansatz, wires=wires, name=name)
        return CircuitComponent(ansatz, wires, name)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        ret._parameters, ret._ansatz = children
        ret._wires, ret._name = aux_data

        # make sure the ansatz parameters match the parameter set
        for param_name, param in ret.ansatz._kwargs.items():
            if isinstance(param, Variable):
                ret.ansatz._kwargs[param_name] = ret.parameters.all_parameters[param.name]
            else:  # need this to build pytree of labels
                ret.ansatz._kwargs[param_name] = param

        return ret

    def auto_shape(self, **_) -> tuple[int, ...]:
        r"""
        The shape of the Fock representation of this component. If the component has a Fock representation
        then it is just the shape of the array. If the component is a ``State`` in Bargmann
        then the shape is calculated using ``autoshape`` using single-mode marginals.
        If the component is not a ``State`` then the shape is a tuple of ``settings.DEFAULT_FOCK_SIZE``
        values except where the ``manual_shape`` attribute has been set.
        """
        return tuple(s or settings.DEFAULT_FOCK_SIZE for s in self.manual_shape)

    def bargmann_triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The Bargmann parametrization of this component, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this component is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`

        If ``batched`` is ``False`` (default), it removes the batch dimension if it is of size 1.

        .. code-block:: pycon

            >>> from mrmustard.lab import CircuitComponent, Coherent
            >>> coh = Coherent(mode=0, x=1.0)
            >>> coh_cc = CircuitComponent.from_bargmann(coh.bargmann_triple(), modes_out_ket=(0,))
            >>> assert isinstance(coh_cc, CircuitComponent)
            >>> assert coh == coh_cc  # equality looks at representation and wires
        """
        try:
            return self.ansatz.triple
        except AttributeError as e:
            raise AttributeError("No Bargmann data for this component.") from e

    def contract(
        self,
        other: CircuitComponent | Scalar,
        mode: str = "kron",
    ) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` without adding adjoints.
        It allows for contracting components exactly as specified.

        For example, a coherent state can be input to an attenuator, but
        the attenuator has two inputs: on the ket and the bra side.
        The ``>>`` operator would automatically add the adjoint of the coherent
        state on the bra side of the input of the attenuator, but the ``@`` operator
        instead does not.

        Args:
            other: The other component to contract with.
            mode: The mode of contraction. Can either "zip" the batch dimensions, "kron" the batch dimensions,
                or pass a custom einsum-style batch string like "ab,cb->ac".

        Returns:
            The contracted component.

        .. code-block::

            >>> from mrmustard.lab import Coherent, Attenuator
            >>> coh = Coherent(0, 1.0)
            >>> att = Attenuator(0, 0.5)
            >>> assert coh.contract(att).wires.input.bra  # the input bra is still uncontracted
        """
        if isinstance(other, numbers.Number | np.ndarray):
            return self * other

        if type(self.ansatz) is not type(other.ansatz):
            if settings.DEFAULT_REPRESENTATION == "Bargmann":
                self_rep = self.to_bargmann()
                other_rep = other.to_bargmann()
            else:
                self_shape = list(self.auto_shape())
                other_shape = list(other.auto_shape())
                contracted_idxs = self.wires.contracted_indices(other.wires)
                for idx1, idx2 in zip(*contracted_idxs):
                    max_shape = max(self_shape[idx1], other_shape[idx2])
                    self_shape[idx1] = max_shape
                    other_shape[idx2] = max_shape
                self_rep = self.to_fock(tuple(self_shape))
                other_rep = other.to_fock(tuple(other_shape))
        else:
            self_rep = self
            other_rep = other

        self_ansatz, self_wires = self_rep.ansatz, self_rep.wires
        other_ansatz, other_wires = other_rep.ansatz, other_rep.wires

        wires_result, _ = self_wires @ other_wires
        core1, core2, core_out = self_wires.contracted_labels(other_wires)
        if mode == "zip":
            eins_str = zip_batch_strings(
                self_ansatz.batch_dims - self_ansatz._lin_sup,
                other_ansatz.batch_dims - other_ansatz._lin_sup,
            )
        elif mode == "kron":
            eins_str = outer_product_batch_str(
                self_ansatz.batch_dims - self_ansatz._lin_sup,
                other_ansatz.batch_dims - other_ansatz._lin_sup,
            )
        else:
            eins_str = mode
        batch12, batch_out = eins_str.split("->")
        batch1, batch2 = batch12.split(",")
        ansatz = self_ansatz.contract(
            other_ansatz,
            list(batch1) + core1,
            list(batch2) + core2,
            list(batch_out) + core_out,
        )
        return CircuitComponent(ansatz, wires_result)

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        r"""
        Returns an array representation of this component in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is generated via ``auto_shape``.
        Returns:
            array: The Fock representation of this component.
        """
        shape = shape or self.auto_shape()
        num_vars = (
            self.ansatz.num_CV_vars
            if isinstance(self.ansatz, PolyExpAnsatz)
            else self.ansatz.num_vars
        )
        if isinstance(shape, int):
            shape = (shape,) * num_vars
        shape = tuple(shape)
        if len(shape) != num_vars:
            raise ValueError(f"Expected Fock shape of length {num_vars}, got {len(shape)}")
        try:
            A, b, c = self.ansatz.triple
            G = math.hermite_renormalized(
                A,
                b,
                math.ones(self.ansatz.batch_shape, dtype=math.complex128),
                shape=shape + self.ansatz.shape_derived_vars,
            )
            G = math.reshape(G, self.ansatz.batch_shape + shape + (-1,))
            cs = math.reshape(c, (*self.ansatz.batch_shape, -1))
            core_str = "".join(
                [chr(i) for i in range(97, 97 + len(G.shape[self.ansatz.batch_dims :]))],
            )
            ret = math.einsum(f"...{core_str},...{core_str[-1]}->...{core_str[:-1]}", G, cs)
            if self.ansatz._lin_sup:
                ret = math.sum(ret, axis=self.ansatz.batch_dims - 1)
        except AttributeError:
            ret = self.ansatz.reduce(shape).array
        return ret

    def on(self, modes: int | Sequence[int]) -> CircuitComponent:
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
        modes = (modes,) if isinstance(modes, int) else modes
        ob = self.wires.output.bra.modes
        ib = self.wires.input.bra.modes
        ok = self.wires.output.ket.modes
        ik = self.wires.input.ket.modes
        subsets = [s for s in (ob, ib, ok, ik) if s]
        if any(s != subsets[0] for s in subsets):
            raise ValueError(
                f"Cannot rewire a component with wires on different modes ({ob, ib, ok, ik}).",
            )
        for subset in subsets:
            if subset and len(subset) != len(modes):
                raise ValueError(f"Expected ``{len(modes)}`` modes, found ``{len(subset)}``.")
        return self._light_copy(
            Wires(
                modes_out_bra=set(modes) if ob else set(),
                modes_in_bra=set(modes) if ib else set(),
                modes_out_ket=set(modes) if ok else set(),
                modes_in_ket=set(modes) if ik else set(),
            ),
        )

    def quadrature(self, *quad: RealVector, phi: float = 0.0) -> ComplexTensor:
        r"""
        The (discretized) quadrature basis representation of the circuit component.
        This method considers the same basis in all the wires. For more fine-grained control,
        use the BtoQ transformation or a combination of transformations.

        Args:
            quad: discretized quadrature points to evaluate over in the
                quadrature representation. One vector of points per wire.
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            A circuit component with the given quadrature representation.
        """
        if isinstance(self.ansatz, ArrayAnsatz):
            conjugates = [i not in self.wires.ket.indices for i in range(len(self.wires.indices))]
            dims = self.ansatz.core_dims

            if len(quad) != dims:
                raise ValueError(
                    f"The fock array has dimension {dims} whereas ``quad`` has {len(quad)}.",
                )
            # construct quadrature basis vectors
            shapes = self.ansatz.core_shape
            quad_basis_vecs = []
            for dim in range(dims):
                q_to_n = oscillator_eigenstate(quad[dim], shapes[dim])
                if not math.allclose(phi, 0.0):
                    theta = -math.arange(shapes[dim]) * phi
                    Ur = math.make_complex(math.cos(theta), math.sin(theta))
                    q_to_n = math.einsum("n,nq->nq", Ur, q_to_n)
                if conjugates[dim]:
                    q_to_n = math.conj(q_to_n)
                quad_basis_vecs += [math.cast(q_to_n, "complex128")]

            # Convert each dimension to quadrature
            fock_string = "".join([chr(97 + self.n_modes + dim) for dim in range(dims)])
            q_string = "".join(
                [
                    f"{fock_string[idx]}{chr(97 + wire.mode)},"
                    for idx, wire in enumerate(self.wires)
                ],
            )[:-1]
            out_string = "".join([chr(97 + mode) for mode in self.modes])
            ret = math.einsum(
                "..." + fock_string + "," + q_string + "->" + out_string + "...",
                self.ansatz.array,
                *quad_basis_vecs,
                optimize=True,
            )
        else:
            batch_str = (
                "".join([chr(97 + wire.mode) + "," for wire in self.wires])[:-1]
                + "->"
                + "".join([chr(97 + mode) for mode in self.modes])
            )
            ret = self.to_quadrature(phi=phi).ansatz.eval(*quad, batch_string=batch_str)
        batch_shape = (
            self.ansatz.batch_shape[:-1] if self.ansatz._lin_sup else self.ansatz.batch_shape
        )
        batch_dims = len(batch_shape)
        size = int(math.prod(ret.shape[:-batch_dims] if batch_shape else ret.shape))
        return math.reshape(ret, (size, *batch_shape))

    def quadrature_triple(
        self,
        phi: float = 0.0,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The quadrature representation triple A,b,c of this circuit component.

        Args:
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            A,b,c triple of the quadrature representation
        """
        return self.to_quadrature(phi=phi).ansatz.triple

    def to_bargmann(self) -> CircuitComponent:
        r"""
        Returns a new ``CircuitComponent`` in the ``Bargmann`` representation.

        .. code-block::

            >>> from mrmustard.lab import Number
            >>> from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz

            >>> num = Number(0, n=2)
            >>> assert isinstance(num.ansatz, ArrayAnsatz) # in Fock representation

            >>> num_bargmann = num.to_bargmann()
            >>> assert isinstance(num_bargmann.ansatz, PolyExpAnsatz) # in Bargmann representation
        """
        if isinstance(self.ansatz, PolyExpAnsatz):
            return self

        if self.ansatz._original_abc_data:
            A, b, c = self.ansatz._original_abc_data
        else:
            A, b, _ = identity_Abc(len(self.wires.quantum))
            c = self.ansatz.data
        ansatz = PolyExpAnsatz(A, b, c)
        wires = self.wires.copy()
        for w in wires.quantum:
            w.repr = ReprEnum.BARGMANN

        cls = type(self)
        params = signature(cls).parameters
        if "mode" in params or "modes" in params:
            ret = self.__class__(self.modes, **self.parameters.to_dict())
            ret._ansatz = ansatz
            ret._wires = wires
        else:
            ret = self._from_attributes(ansatz, wires, self.name)
        return ret

    def to_fock(self, shape: int | Sequence[int] | None = None) -> CircuitComponent:
        r"""
        Returns a new ``CircuitComponent`` in the ``Fock`` representation.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as
                an ``int``, it is broadcasted to all dimensions. If ``None``, it
                is generated via ``auto_shape``.

        Returns:
            A new ``CircuitComponent`` in the ``Fock`` representation.

        .. code-block::

            >>> from mrmustard.lab import Dgate
            >>> from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz

            >>> d = Dgate(1, x=0.1, y=0.1)
            >>> assert isinstance(d.ansatz, PolyExpAnsatz) # in Bargmann representation

            >>> d_fock = d.to_fock(shape=3)
            >>> assert isinstance(d_fock.ansatz, ArrayAnsatz) # in Fock representation
        """
        shape = shape or self.auto_shape()
        batch_dims = self.ansatz.batch_dims - 1 if self.ansatz._lin_sup else self.ansatz.batch_dims
        fock = ArrayAnsatz(self.fock_array(shape), batch_dims=batch_dims)
        try:
            if self.ansatz.num_derived_vars == 0:
                fock._original_abc_data = self.ansatz.triple
        except AttributeError:
            fock._original_abc_data = None
        wires = self.wires.copy()
        for w in wires.quantum:
            w.repr = ReprEnum.FOCK
            w.fock_cutoff = fock.core_shape[w.index]

        cls = type(self)
        params = signature(cls).parameters
        if "mode" in params or "modes" in params:
            ret = self.__class__(self.modes, **self.parameters.to_dict())
            ret._ansatz = fock
            ret._wires = wires
        else:
            ret = self._from_attributes(fock, wires, self.name)
        return ret

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
        from .circuit_components_utils.b_to_q import BtoQ  # noqa: PLC0415

        BtoQ_ob = BtoQ(self.wires.output.bra.modes, phi).adjoint
        BtoQ_ib = BtoQ(self.wires.input.bra.modes, phi).adjoint.dual
        BtoQ_ok = BtoQ(self.wires.output.ket.modes, phi)
        BtoQ_ik = BtoQ(self.wires.input.ket.modes, phi).dual

        object_to_convert = self
        if isinstance(self.ansatz, ArrayAnsatz):
            object_to_convert = self.to_bargmann()

        return BtoQ_ib.contract(BtoQ_ik.contract(object_to_convert).contract(BtoQ_ok)).contract(
            BtoQ_ob,
        )

    def _light_copy(self, wires: Wires | None = None) -> CircuitComponent:
        r"""
        Creates a "light" copy of this component by referencing its __dict__, except for the wires,
        which are a new object or the given one.
        This is useful when one needs the same component acting on different modes, for example.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance._ansatz = self.ansatz
        instance._wires = wires or Wires(*self.wires.args)
        return instance

    def _rshift_return(
        self,
        result: CircuitComponent | np.ndarray | complex,
    ) -> CircuitComponent | np.ndarray | complex:
        "internal convenience method for right-shift, to return the right type of object"
        if len(result.wires) > 0:
            return result
        return result.ansatz.scalar

    def _serialize(self) -> tuple[dict[str, Any], dict[str, ArrayLike]]:
        """
        Inner serialization to be used by Circuit.serialize().

        The first dict must be JSON-serializable, and the second dict must contain
        the (non-JSON-serializable) array-like data to be collected separately.

        Returns:
            A tuple containing the serialized data and the array-like data.
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
        elif "mode" in params:
            serializable["mode"] = tuple(self.wires.modes)
        else:
            raise TypeError(f"{cls.__name__} does not seem to have any wires construction method")

        if self.parameters:
            for k, v in self.parameters.variables.items():
                serializable[f"{k}_bounds"] = v.bounds
                serializable[f"{k}_trainable"] = True
            return serializable, {k: v.value for k, v in self.parameters.all_parameters.items()}

        return serializable, {}

    def _tree_flatten(self):  # pragma: no cover
        children = (self.parameters, self.ansatz)
        aux_data = (self.wires, self.name)
        return (children, aux_data)

    def __add__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Implements the addition between circuit components.
        """
        if self.wires != other.wires:
            raise ValueError("Cannot add components with different wires.")

        if (
            isinstance(self.ansatz, PolyExpAnsatz)
            and self.ansatz._fn is not None
            and self.ansatz._fn == other.ansatz._fn
        ):
            new_params = {}
            for name in self.ansatz._kwargs:
                self_param = getattr(self.parameters, name)
                other_param = getattr(other.parameters, name)
                if (self_type := type(self_param)) is not (other_type := type(other_param)):
                    raise ValueError(
                        f"Parameter '{name}' is a {self_type.__name__} for one component and a {other_type.__name__} for the other."
                    )
                if (self.ansatz.batch_dims - self.ansatz._lin_sup) > 0 or (
                    other.ansatz.batch_dims - other.ansatz._lin_sup
                ) > 0:
                    raise ValueError("Cannot add batched components.")
                if isinstance(self_param, Variable):
                    if self_param.bounds != other_param.bounds:
                        raise ValueError(
                            f"Parameter '{name}' has bounds {self_param.bounds} and {other_param.bounds} for the two components."
                        )
                    new_params[name + "_trainable"] = True
                    new_params[name + "_bounds"] = self_param.bounds
                self_val = math.atleast_nd(self_param.value, 1)
                other_val = math.atleast_nd(other_param.value, 1)
                new_params[name] = math.concat((self_val, other_val), axis=0)
            ret = self.__class__(self.modes, **new_params)
            ret.ansatz._lin_sup = True
            return ret
        ansatz = self.ansatz + other.ansatz
        name = self.name if self.name == other.name else ""
        ret = self._from_attributes(ansatz, self.wires, name)
        ret.manual_shape = tuple(
            max(a, b) if a is not None and b is not None else a or b
            for a, b in zip(self.manual_shape, other.manual_shape)
        )
        return ret

    def __eq__(self, other) -> bool:
        r"""
        Whether this component is equal to another component.

        Compares representations, but not the other attributes
        (e.g. name and parameter set).
        """
        if not isinstance(other, CircuitComponent):
            return False
        return self.ansatz == other.ansatz and self.wires == other.wires

    def __mul__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the multiplication by a scalar from the right.
        """
        return self._from_attributes(self.ansatz * other, self.wires, self.name)

    def __repr__(self) -> str:
        ansatz = self.ansatz
        repr_name = ansatz.__class__.__name__
        if repr_name == "NoneType":
            return self.__class__.__name__ + f"(modes={self.modes}, name={self.name})"
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
        return (self * other).ansatz.scalar

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

            >>> from mrmustard.lab import Coherent, Attenuator, Ket, DM, Channel
            >>> state = Coherent(0, 1.0)
            >>> assert issubclass(Coherent, Ket)
            >>> assert issubclass(Attenuator, Channel)
            >>> assert isinstance(state >> Attenuator(0, 0.5), DM)
            >>> assert math.allclose(state >> state.dual, 1+0j)
        """
        if hasattr(other, "__custom_rrshift__"):
            return other.__custom_rrshift__(self)

        if isinstance(other, numbers.Number | np.ndarray):
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
            ret = self.contract(other)
        elif self_needs_bra or self_needs_ket:
            ret = self.adjoint.contract(self.contract(other), "zip")
        elif other_needs_bra or other_needs_ket:
            ret = self.contract(other.adjoint).contract(other)
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
        return self._from_attributes(ansatz, self.wires, name)

    def __truediv__(self, other: Scalar) -> CircuitComponent:
        r"""
        Implements the division by a scalar for circuit components.
        """
        return self._from_attributes(self.ansatz / other, self.wires, self.name)

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
