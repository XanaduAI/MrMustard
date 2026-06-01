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

"""A base class for the components of quantum circuits."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from mrmustard import math, settings
from mrmustard import widgets as mmwidgets
from mrmustard.parameters import ParameterDict
from mrmustard.physics.ansatz import Ansatz, ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.fock_utils import oscillator_eigenstate
from mrmustard.physics.mm_einsum import bargmann_to_fock, fock_to_bargmann
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexScalar,
    ComplexTensor,
    ComplexVector,
    RealVector,
    Scalar,
)

__all__ = ["CircuitComponent"]


class CircuitComponent:
    r"""A base class for the circuit components (states, transformations, measurements,
    and any component made by combining CircuitComponents). CircuitComponents are
    defined by their ``ansatz`` and ``wires``.

    Args:
        ansatz_factory: The AnsatzFactory of this circuit component.
        wires: The wires of this circuit component.
        name: The name of this circuit component.
    """

    short_name = "CC"

    def __init__(
        self,
        ansatz_factory: AnsatzFactory | None = None,
        wires: Wires | None = None,
        name: str | None = None,
    ) -> None:
        self._ansatz_factory = ansatz_factory
        self._name = name
        self._parameters = ParameterDict()
        self._wires = wires or Wires(set(), set(), set(), set())

    @property
    def adjoint(self) -> CircuitComponent:
        r"""The adjoint of this component obtained by conjugating the ansatz and swapping
        the ket and bra wires.

        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0])
        >>> assert psi.dm() == psi.contract(psi.adjoint)

        Note:
            The resulting CircuitComponent is not in standard order.
            To get the standard order, call ``to_standard_order()`` on the result.
        """
        ansatz_factory, _ = AnsatzFactory.from_ansatz(self.ansatz.conj)
        name = self.name.removesuffix("_adj") if self.name.endswith("_adj") else self.name + "_adj"
        ret = CircuitComponent(ansatz_factory=ansatz_factory, wires=self.wires.adjoint, name=name)
        ret.short_name = self.short_name
        ret._parameters = self.parameters.copy()
        return ret

    @property
    def ansatz(self) -> Ansatz:
        r"""The ansatz of this circuit component."""
        representation = ReprEnum.BARGMANN
        shape = ()
        if len(self.wires.quantum) != 0:
            for w in self.wires.quantum:
                if w.repr == ReprEnum.FOCK:
                    representation = ReprEnum.FOCK
                shape += (w.fock_shape,)
        elif self.ansatz_factory.ansatz_dict.get(ReprEnum.BARGMANN, None) is None:
            representation = ReprEnum.FOCK
        return self.ansatz_factory(**self.parameters, representation=representation, shape=shape)

    @property
    def ansatz_factory(self) -> AnsatzFactory:
        r"""The ansatz factory of this component.

        Raises:
            AttributeError: If the component has no ``AnsatzFactory``.
        """
        if self._ansatz_factory is None:
            raise AttributeError("CircuitComponent has no ansatz factory.")
        return self._ansatz_factory

    @property
    def dual(self) -> CircuitComponent:
        r"""The dual of this component obtained by conjugating the ansatz and swapping
        the input and output wires.

        >>> from mrmustard import math
        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0])
        >>> assert math.allclose(1.0, psi >> psi.dual)

        Note:
            The resulting CircuitComponent is not in standard order.
            To get the standard order, call ``to_standard_order()`` on the result.
        """
        ansatz_factory, _ = AnsatzFactory.from_ansatz(self.ansatz.conj)
        name = (
            self.name.removesuffix("_dual") if self.name.endswith("_dual") else self.name + "_dual"
        )
        ret = CircuitComponent(ansatz_factory=ansatz_factory, wires=self.wires.dual, name=name)
        ret.short_name = self.short_name
        ret._parameters = self.parameters.copy()
        return ret

    @property
    def manual_shape(self) -> tuple[int | None, ...]:
        r"""The shape of this Component in the Fock representation in standard order. For each wire,
        the entry is either an integer or ``None``. If it is an integer, it
        is the shape (dimension) of the corresponding Fock space. For a cutoff of n
        (maximum photon number n), the shape is n + 1
        (includes :math:`\left\vert 0\right\rangle` through :math:`\left\vert n\right\rangle`).
        If it is ``None``, it means the best shape is not known yet and will be determined by
        ``auto_shape``.
        """
        return self.wires.fock_shapes

    @manual_shape.setter
    def manual_shape(self, shape: tuple[int | None, ...]):
        for w, s in zip(self.wires, shape):
            w.fock_shape = s

    @property
    def modes(self) -> tuple[int, ...]:
        r"""The sorted tuple of modes of this component.

        >>> from mrmustard.lab import GaussianKet
        >>> ket = GaussianKet.random([0, 1])
        >>> assert ket.modes == (0, 1)
        """
        return tuple(sorted(self.wires.modes))

    @property
    def name(self) -> str:
        r"""The name of this component.

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
        r"""The number of modes spanned by this component across all wires.

        >>> from mrmustard.lab import GaussianKet
        >>> ket = GaussianKet.random([0, 1])
        >>> assert ket.n_modes == 2
        """
        return len(self.modes)

    @property
    def parameters(self) -> ParameterDict:
        r"""The parameters of this component."""
        return self._parameters

    @property
    def wires(self) -> Wires:
        r"""The wires of this circuit component."""
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
        r"""Initializes a ``CircuitComponent`` object from its Bargmann (A,b,c) parametrization.

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
        return cls._from_attributes(ansatz, wires, name)

    @classmethod
    def from_fock(
        cls,
        array: ComplexTensor,
        batch_dims: int,
        modes_out_bra: Sequence[int] = (),
        modes_in_bra: Sequence[int] = (),
        modes_out_ket: Sequence[int] = (),
        modes_in_ket: Sequence[int] = (),
        name: str | None = None,
    ) -> CircuitComponent:
        r"""Initializes a ``CircuitComponent`` object from its Fock parametrization.

        Args:
            array: The Fock array of the component.
            batch_dims: The number of batch dimensions in the given array.
            modes_out_bra: The output modes on the bra side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_ket: The input modes on the ket side of this component.
            name: The name of this component.

        Returns:
            A ``CircuitComponent`` with the given Fock representation.

        Raises:
            ValueError: If the given array has a shape that is inconsistent with the number of modes.
        """
        expected_core_dims = (
            len(modes_out_bra) + len(modes_in_bra) + len(modes_out_ket) + len(modes_in_ket)
        )
        num_core_dims = array.ndim - batch_dims
        if num_core_dims != expected_core_dims:
            raise ValueError(
                f"Wires expect {expected_core_dims} core dimensions, got {num_core_dims} in the given array."
            )

        ansatz = ArrayAnsatz(array, batch_dims=batch_dims)
        wires = Wires(set(modes_out_bra), set(modes_in_bra), set(modes_out_ket), set(modes_in_ket))
        for w in wires.quantum:
            w.repr = ReprEnum.FOCK
            w.fock_shape = ansatz.core_shape[w.index]
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
        r"""Returns a circuit component from the given triple (A,b,c) that parametrizes the
        quadrature wavefunction of this component in the form :math:`c * exp(1/2 x^T A x + b^T x)`.

        Args:
            modes_out_bra: The output modes on the bra side of this component.
            modes_in_bra: The input modes on the bra side of this component.
            modes_out_ket: The output modes on the ket side of this component.
            modes_in_ket: The input modes on the ket side of this component.
            triple: The (A,b,c) triple that parametrizes the wave function.
            phi: The quadrature angle. ``0`` corresponds to the x quadrature, ``pi/2`` to the p quadrature.
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
        ansatz_factory, _ = AnsatzFactory.from_ansatz(PolyExpAnsatz(*triple), ReprEnum.BARGMANN)
        # NOTE: the representation is Bargmann here because we use the inverse of BtoQ on the B side
        QQQQ = CircuitComponent(ansatz_factory=ansatz_factory, wires=wires)
        BBBB = QtoB_ib.contract(QtoB_ik.contract(QQQQ).contract(QtoB_ok)).contract(QtoB_ob)
        return cls._from_attributes(BBBB.ansatz, wires, name)

    @classmethod
    def _from_attributes(
        cls,
        ansatz: Ansatz | None,
        wires: Wires,
        name: str | None = None,
    ) -> CircuitComponent:
        r"""Initializes a circuit component from an ``Ansatz``, ``Wires`` and a name.

        It differs from the __init__ in that the return type is the closest parent
        among the types ``Ket``, ``DM``, ``Unitary``, ``Operation``, ``Channel``,
        and ``Map``. This is to ensure the right properties are used when calling
        methods on the returned object, e.g. when adding two coherent states we
        don't get a generic ``CircuitComponent`` but a ``Ket``:

        >>> from mrmustard.lab import Coherent, Ket
        >>> cat = Coherent(mode=0, alpha=2.0) + Coherent(mode=0, alpha=-2.0)
        >>> assert isinstance(cat, Ket)

        Args:
            ansatz: An ansatz for this circuit component.
            wires: The wires for this circuit component.
            name: The name for this component (optional).

        Returns:
            A circuit component with the given attributes.
        """
        if ansatz is not None:
            ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz)
            if representation == ReprEnum.FOCK:
                for w in wires.quantum:
                    w.repr = ReprEnum.FOCK
                    w.fock_shape = ansatz.core_shape[w.index]
        else:
            ansatz_factory = None
        types = {"Ket", "DM", "Unitary", "Operation", "Channel", "Map"}
        for tp in cls.mro():
            if tp.__name__ in types:
                return tp(ansatz_factory=ansatz_factory, wires=wires, name=name)
        return CircuitComponent(ansatz_factory=ansatz_factory, wires=wires, name=name)

    def auto_shape(self, **_) -> tuple[int, ...]:
        r"""The shape of the Fock representation of this component. If the component has a Fock representation
        then it is just the shape of the array. If the component is a ``State`` in Bargmann
        then the shape is calculated using ``autoshape`` using single-mode marginals.
        If the component is not a ``State`` then the shape is a tuple of ``settings.DEFAULT_FOCK_SIZE``
        values except where the ``manual_shape`` attribute has been set.
        """
        return tuple(s or settings.DEFAULT_FOCK_SIZE for s in self.manual_shape)

    def bargmann_triple(
        self,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexScalar]:
        r"""The Bargmann parametrization of this component, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this component is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`.

        >>> from mrmustard.lab import CircuitComponent, Coherent
        >>> coh = Coherent(mode=0, alpha=1.0)
        >>> coh_cc = CircuitComponent.from_bargmann(coh.bargmann_triple(), modes_out_ket=(0,))
        >>> assert isinstance(coh_cc, CircuitComponent)
        >>> assert coh == coh_cc  # equality looks at representation and wires

        Returns:
            The Bargmann triple of this component.

        Raises:
            AttributeError: If the component has no Bargmann data.
        """
        try:
            return self.ansatz_factory(
                **self.parameters, representation=ReprEnum.BARGMANN, shape=self.manual_shape
            ).triple
        except (NotImplementedError, AttributeError) as e:
            raise AttributeError("No Bargmann data for this component.") from e

    def contract(self, other: CircuitComponent | Scalar) -> CircuitComponent:
        r"""Contracts ``self`` and ``other`` without adding adjoints.
        Core index selection is determined solely by the wires; batch dimensions
        are broadcast automatically by the underlying ansatz implementations.

        For example, a coherent state can be input to an attenuator, but the
        attenuator has two inputs: on the ket and the bra side. The ``>>`` operator
        would automatically add the adjoint of the coherent state on the bra side
        of the input of the attenuator, but ``contract`` instead does not.

        >>> from mrmustard.lab import Coherent, Attenuator
        >>> coh = Coherent(0, 1.0)
        >>> att = Attenuator(0, 0.5)
        >>> assert coh.contract(att).wires.input.bra  # the input bra is still uncontracted

        Args:
            other: The other component to contract with.

        Returns:
            The contracted component.
        """
        if isinstance(other, (numbers.Number, np.ndarray)):
            return self * other

        if (self_type := type(self.ansatz)) is not (other_type := type(other.ansatz)):
            if settings.DEFAULT_REPRESENTATION is None:
                raise TypeError(
                    f"{self_type=} does not match {other_type=} and "
                    "DEFAULT_REPRESENTATION is None. You must be more explicit to "
                    "perform this contraction by calling `to_bargmann() or `to_fock()."
                )
            if settings.DEFAULT_REPRESENTATION == "Bargmann":
                self_rep = self.to_bargmann()
                other_rep = other.to_bargmann()
                representation = ReprEnum.BARGMANN
            else:
                self_shape, other_shape = self._fock_shapes_for_contraction(other)
                self_rep = self.to_fock(tuple(self_shape))
                other_rep = other.to_fock(tuple(other_shape))
                representation = ReprEnum.FOCK
        else:
            self_rep = self
            other_rep = other
            representation = None

        wires_result, core_perm = self_rep.wires @ other_rep.wires
        core1, core2 = self_rep.wires.contracted_indices(other_rep.wires)
        ansatz = self_rep.ansatz.contract(other_rep.ansatz, (core1, core2)).reorder(core_perm)
        ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz, representation)
        if representation == ReprEnum.FOCK:
            for w in wires_result.quantum:
                w.repr = ReprEnum.FOCK
                w.fock_shape = ansatz.core_shape[w.index]
        return CircuitComponent(ansatz_factory=ansatz_factory, wires=wires_result)

    def concat(self, other: CircuitComponent, axis: int = 0) -> CircuitComponent:
        r"""Concatenates this component with another along the specified batch axis.

        The ansatze are concatenated along the given batch axis. The wires must match.
        Returns an instance of the closest common superclass.

        >>> from mrmustard.lab import Coherent
        >>> state1 = Coherent(mode=0, alpha=1.0)
        >>> state2 = Coherent(mode=0, alpha=2.0)
        >>> # Add batch dimension first
        >>> state1_batched = state1[None]
        >>> state2_batched = state2[None]
        >>> concatenated = state1_batched.concat(state2_batched, axis=0)
        >>> assert concatenated.ansatz.batch_shape == (2,)

        Args:
            other: The other circuit component to concatenate with.
            axis: The batch axis along which to concatenate.

        Returns:
            A new CircuitComponent with concatenated ansatze.

        Raises:
            ValueError: If wires don't match or if ansatz types don't match.
        """
        if (sw := self.wires) != (ow := other.wires):
            raise ValueError(f"Can't stack or concat components with different wires: {sw} vs {ow}")

        if (sa := type(self.ansatz)) is not (oa := type(other.ansatz)):
            raise ValueError(
                f"Can't stack or concat components with different ansatz types: {sa} vs {oa}"
            )

        concat_ansatz = self.ansatz.concat(other.ansatz, axis=axis)

        return self._from_attributes(concat_ansatz, self.wires, self.name)

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        r"""Returns an array representation of this component in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is generated via ``auto_shape``.

        Returns:
            array: The Fock representation of this component.

        Raises:
            ValueError: If the shape is not valid for the component.
        """
        shape = self._check_fock_shape(shape)
        ansatz_factory = self.ansatz_factory
        if ansatz_factory.ansatz_dict.get(ReprEnum.FOCK, None) is None:
            ansatz_factory = self.to_fock(shape).ansatz_factory
        return (
            ansatz_factory(
                **self.parameters,
                representation=ReprEnum.FOCK,
                shape=shape,
            )
            .reduce(shape)
            .array
        )

    def _fock_shapes_for_contraction(
        self, other: CircuitComponent
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""Helper function to get the optimal shapes for contracting self with other in Fock."""
        self_shape = list(self.auto_shape())
        other_shape = list(other.auto_shape())
        contracted_idxs = self.wires.contracted_indices(other.wires)
        for idx1, idx2 in zip(*contracted_idxs):
            min_shape = min(self_shape[idx1], other_shape[idx2])
            override = self.manual_shape[idx1] or other.manual_shape[idx2]
            override_shape = max(self.manual_shape[idx1] or 1, other.manual_shape[idx2] or 1)
            self_shape[idx1] = min_shape if not override else override_shape
            other_shape[idx2] = min_shape if not override else override_shape
        return tuple(self_shape), tuple(other_shape)

    def on(self, modes: int | Sequence[int]) -> CircuitComponent:
        r"""Creates a light copy of this component that acts on the given ``modes`` instead of the
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
        wires = Wires(
            modes_out_bra=set(modes) if ob else set(),
            modes_in_bra=set(modes) if ib else set(),
            modes_out_ket=set(modes) if ok else set(),
            modes_in_ket=set(modes) if ik else set(),
        )
        for w_o, w_n in zip(self.wires.index_order, wires.index_order):
            w_n.repr = w_o.repr
            w_n.fock_shape = w_o.fock_shape
        return self._light_copy(wires=wires)

    def quadrature(self, *quad: RealVector, phi: float = 0.0) -> ComplexTensor:
        r"""The (discretized) quadrature basis representation of the circuit component.
        This method considers the same basis in all the wires. For more fine-grained control,
        use the BtoQ transformation or a combination of transformations.

        Args:
            quad: discretized quadrature points to evaluate over in the
                quadrature representation. One vector of points per wire.
            phi: The quadrature angle. ``0`` corresponds to the x quadrature, ``pi/2`` to the p quadrature.

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

            # Assign compact einsum characters: first modes, then fock dims.
            mode_char = {m: chr(97 + i) for i, m in enumerate(self.modes)}
            fock_string = "".join(chr(97 + self.n_modes + d) for d in range(dims))
            q_string = ",".join(
                f"{fock_string[idx]}{mode_char[wire.mode]}" for idx, wire in enumerate(self.wires)
            )
            out_string = "".join(mode_char[mode] for mode in self.modes)
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
        size = 1
        for i in ret.shape[:-batch_dims] if batch_shape else ret.shape:
            size *= i
        return math.reshape(ret, (size, *batch_shape))

    def quadrature_triple(
        self,
        phi: float = 0.0,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexScalar]:
        r"""The quadrature representation triple A,b,c of this circuit component.

        Args:
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature.

        Returns:
            A,b,c triple of the quadrature representation
        """
        return self.to_quadrature(phi=phi).ansatz.triple

    def stack(self, other: CircuitComponent, axis: int = 0) -> CircuitComponent:
        r"""Stacks this component with another along a new batch axis.

        The two components must have the same wires and compatible ansatze.
        A new batch axis is inserted at the specified position.

        >>> from mrmustard.lab import Coherent
        >>> state1 = Coherent(mode=0, alpha=1.0)
        >>> state2 = Coherent(mode=0, alpha=2.0)
        >>> stacked = state1.stack(state2, axis=0)
        >>> assert stacked.ansatz.batch_shape == (2,)

        Args:
            other: The other circuit component to stack with.
            axis: The position where the new batch axis will be inserted.

        Returns:
            A new CircuitComponent with stacked ansatze.

        Raises:
            ValueError: If wires don't match, ansatz types don't match, or axis is out of range.
        """
        if (sw := self.wires) != (ow := other.wires):
            raise ValueError(f"Can't stack components with different wires: {sw} vs {ow}")

        if (sa := type(self.ansatz)) is not (oa := type(other.ansatz)):
            raise ValueError(f"Can't stack components with different ansatz types: {sa} vs {oa}")

        # Normalize negative axis
        if axis < 0:
            axis = self.ansatz.batch_dims + 1 + axis

        if axis < 0 or axis > self.ansatz.batch_dims:
            raise ValueError(f"axis {axis} is out of range for batch_dims {self.ansatz.batch_dims}")

        # Add a new axis at the specified position to both components
        index_tuple = (
            (slice(None),) * axis + (None,) + (slice(None),) * (self.ansatz.batch_dims - axis)
        )
        self_expanded = self[index_tuple]
        other_expanded = other[index_tuple]

        # Concatenate along the new axis
        return self_expanded.concat(other_expanded, axis=axis)

    def to_bargmann(self) -> CircuitComponent:
        r"""Returns a new ``CircuitComponent`` in the ``Bargmann`` representation.

        >>> from mrmustard.lab import Dgate, Number
        >>> from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
        >>> d = Dgate(1, alpha=0.1 + 0.1j)
        >>> d_fock = d.to_fock(shape=3)
        >>> d_bargmann = d_fock.to_bargmann()
        >>> num = Number(0, n=2)
        >>> assert isinstance(num.ansatz, ArrayAnsatz) # in Fock representation
        >>> num_bargmann = num.to_bargmann()
        >>> assert isinstance(num_bargmann.ansatz, PolyExpAnsatz) # in Bargmann representation

        Returns:
            A new ``CircuitComponent`` in the ``Bargmann`` representation.
        """
        ansatz_factory = self.ansatz_factory
        representations = set(self.wires.representations)
        if len(representations) == 1 and ReprEnum.BARGMANN in representations:
            return self

        wires = self.wires.copy()
        for w in wires.quantum:
            w.repr = ReprEnum.BARGMANN
        ret = self._light_copy(wires=wires)

        ansatz_dict = ansatz_factory.ansatz_dict
        if ansatz_dict.get(ReprEnum.BARGMANN, None) is None:
            fock_fn, fock_params = ansatz_dict[ReprEnum.FOCK]
            ansatz_dict[ReprEnum.BARGMANN] = (fock_to_bargmann(fock_fn), fock_params)
        else:
            shape = self.manual_shape
            bargmann_ansatz = self.ansatz_factory(
                **self.parameters, representation=ReprEnum.BARGMANN, shape=shape
            )
            if bargmann_ansatz.num_derived_vars > 0 and shape != bargmann_ansatz.shape_derived_vars:
                ansatz_factory, _ = AnsatzFactory.from_ansatz(self.ansatz, ReprEnum.FOCK)
                ansatz_dict = ansatz_factory.ansatz_dict
                fock_fn, fock_params = ansatz_dict[ReprEnum.FOCK]
                ansatz_dict[ReprEnum.BARGMANN] = (
                    fock_to_bargmann(fock_fn),
                    (*fock_params, "shape"),
                )
                ret._ansatz_factory = ansatz_factory

        return ret

    def to_fock(self, shape: int | Sequence[int] | None = None) -> CircuitComponent:
        r"""Returns a new ``CircuitComponent`` in the ``Fock`` representation.

        >>> from mrmustard.lab import Dgate
        >>> from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
        >>> d = Dgate(1, alpha=0.1 + 0.1j)
        >>> d_fock = d.to_fock(shape=3)
        >>> assert d_fock.name == d.name
        >>> assert isinstance(d.ansatz, PolyExpAnsatz) # in Bargmann representation
        >>> assert isinstance(d_fock.ansatz, ArrayAnsatz) # in Fock representation

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as
                an ``int``, it is broadcasted to all dimensions. If ``None``, it
                is generated via ``auto_shape``.

        Returns:
            A new ``CircuitComponent`` in the ``Fock`` representation.
        """
        shape = self._check_fock_shape(shape)
        ansatz_factory = self.ansatz_factory
        if (ansatz_dict := ansatz_factory.ansatz_dict).get(ReprEnum.FOCK, None) is None:
            bargmann_fn, bargmann_params = ansatz_dict[ReprEnum.BARGMANN]
            ansatz_dict[ReprEnum.FOCK] = (
                bargmann_to_fock(bargmann_fn),
                (*bargmann_params, "shape"),
            )
        wires = self.wires.copy()
        for w in wires.quantum:
            w.repr = ReprEnum.FOCK
            w.fock_shape = shape[w.index]
        return self._light_copy(wires=wires)

    def to_quadrature(self, phi: float = 0.0) -> CircuitComponent:
        r"""Returns a circuit component with the quadrature representation of this component
        in terms of A,b,c.

        Args:
            phi: The quadrature angle. ``0`` corresponds to the x quadrature, ``pi/2`` to the p quadrature.

        Returns:
            A circuit component with the given quadrature representation.
        """
        from .circuit_components_utils.b_to_q import BtoQ  # noqa: PLC0415

        BtoQ_ob = BtoQ(self.wires.output.bra.modes, phi).adjoint
        BtoQ_ib = BtoQ(self.wires.input.bra.modes, phi).adjoint.dual
        BtoQ_ok = BtoQ(self.wires.output.ket.modes, phi)
        BtoQ_ik = BtoQ(self.wires.input.ket.modes, phi).dual

        object_to_convert = self.to_bargmann()

        return BtoQ_ib.contract(BtoQ_ik.contract(object_to_convert).contract(BtoQ_ok)).contract(
            BtoQ_ob,
        )

    def to_standard_order(self) -> CircuitComponent:
        r"""Reorders the ansatz and wires to the standard order.

        Returns:
            A circuit component in the standard order.
        """
        # make a copy of the wires and reindex them
        wires_standard_order = self.wires.copy(new_ids=True)
        wires_standard_order._reindex()

        # create a new instance of the component with the reindexed wires
        ret = self._light_copy(wires_standard_order)

        # reorder the ansatz to the standard order
        ok = self.wires.ket.output.indices
        ik = self.wires.ket.input.indices
        ib = self.wires.bra.input.indices
        ob = self.wires.bra.output.indices
        ansatz_standard_order = self.ansatz.reorder(ob + ib + ok + ik)

        # update the ansatz factory
        ret._ansatz_factory, _ = AnsatzFactory.from_ansatz(ansatz_standard_order)
        return ret

    def _check_fock_shape(self, shape: int | Sequence[int] | None = None) -> tuple[int, ...]:
        r"""Checks that the given shape is valid for the component and returns the final Fock shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component.

        Args:
            shape: The Fock shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is generated via ``auto_shape``.

        Returns:
            The shape of the Fock representation of this component.

        Raises:
            ValueError: If the shape either contains 0 or is not the correct length.
        """
        shape = shape if shape is not None else self.auto_shape()
        ansatz = self.ansatz
        num_vars = ansatz.num_CV_vars if isinstance(ansatz, PolyExpAnsatz) else ansatz.num_vars
        if isinstance(shape, (int, math.int64)):
            shape = (shape,) * num_vars
        shape = tuple(shape)
        if 0 in shape:
            raise ValueError(f"Expected a non-zero Fock shape, got {shape}.")
        if len(shape) != num_vars:
            raise ValueError(f"Expected Fock shape of length {num_vars}, got {len(shape)}")
        return shape

    def _light_copy(self, wires: Wires | None = None) -> CircuitComponent:
        r"""Creates a "light" copy of this component by referencing its __dict__, except for the wires,
        which are a new object or the given one.
        This is useful when one needs the same component acting on different modes, for example.
        """
        instance = super().__new__(self.__class__)
        instance.__dict__ = self.__dict__.copy()
        instance._ansatz_factory = self.ansatz_factory
        instance._wires = wires or self.wires.copy()
        return instance

    def _rshift_return(
        self,
        result: CircuitComponent | Scalar,
    ) -> CircuitComponent | Scalar:
        """Internal convenience method for right-shift, to return the right type of object."""
        if len(result.wires) > 0:
            return result
        return result.ansatz.scalar

    def __add__(self, other: CircuitComponent) -> CircuitComponent:
        r"""Implements the addition between circuit components."""
        if self.wires != other.wires:
            raise ValueError("Cannot add components with different wires.")
        ansatz = self.ansatz + other.ansatz
        name = self.name if self.name == other.name else ""
        ret = self._from_attributes(ansatz, self.wires, name)
        ret.manual_shape = tuple(
            max(a, b) if a is not None and b is not None else a or b
            for a, b in zip(self.manual_shape, other.manual_shape)
        )
        return ret

    def __eq__(self, other) -> bool:
        r"""Whether this component is equal to another component.

        Compares representations, but not the other attributes
        (e.g. name and parameter set).
        """
        if not isinstance(other, CircuitComponent):
            return False
        return self.ansatz == other.ansatz and self.wires == other.wires

    def __getitem__(self, index: Any) -> CircuitComponent:
        r"""Batch-only indexing on the underlying ansatz. Mirrors ``Ansatz``'s batch-only ``__getitem__``.
        Core axes and mode selection are not indexable here. Use ``get_modes`` for subsystem selection.

        Raises:
            AttributeError: If the circuit component has no ansatz to index.
        """
        if self.ansatz is None:
            raise AttributeError("This circuit component has no ansatz to index.")
        return self._from_attributes(self.ansatz[index], self.wires, self.name)

    def __mul__(self, other: Scalar) -> CircuitComponent:
        r"""Implements the multiplication by a scalar from the right."""
        return self._from_attributes(self.ansatz * other, self.wires, self.name)

    def __repr__(self) -> str:
        try:
            return (
                self.__class__.__name__
                + f"(modes={self.modes}, name={self.name}"
                + f", repr={self.ansatz!s})"
            )
        except AttributeError as e:
            if str(e) != "CircuitComponent has no ansatz factory.":
                raise e
            return self.__class__.__name__ + f"(modes={self.modes}, name={self.name})"

    def __rmatmul__(self, other: Scalar) -> CircuitComponent:
        r"""Multiplies a scalar with a circuit component when written as ``scalar @ component``."""
        return self * other

    def __rmul__(self, other: Scalar) -> CircuitComponent:
        r"""Implements the multiplication by a scalar from the left."""
        return self * other

    def __rrshift__(self, other: Scalar) -> Scalar:
        r"""Multiplies a scalar with a circuit component when written as ``scalar >> component``.
        This is needed when the "component" on the left is the result of a contraction that leaves
        no wires and the component is returned as a scalar. Note that there is an edge case if the
        object on the left happens to have the ``__rshift__`` method, but it's not the one we want
        (usually `>>` is about bit shifts) like a numpy array. In this case in an expression with
        types ``np.ndarray >> CircuitComponent`` the method ``CircuitComponent.__rrshift__`` will
        not be called, and something else will be returned.
        """
        return (self * other).ansatz.scalar

    def __rshift__(self, other: Scalar | CircuitComponent) -> Scalar | CircuitComponent:
        r"""Contracts ``self`` and ``other`` (output of self going into input of other).
        It adds the adjoints when they are missing (e.g. if ``self`` is a ``Ket`` and
        ``other`` is a ``Channel``). An error is raised if these cannot be deduced from
        the wires of the components. For example this allows ``Ket`` to be right-shifted
        into ``Channel`` and automatically the result is a ``DM``. If the result has
        no wires left, it returns the (batched) scalar value of the representation.
        Note that a ``CircuitComponent`` is allowed to right-shift into scalars because the scalar
        part may result from an automated contraction subroutine that involves several components).

        Note that the resulting component type is coerced based on the wires of the result:

        - ``Ket``: only output ket wires remain
        - ``DM``: only output bra and ket on the same modes remain
        - ``Unitary``: only bra wires remain
        - ``Channel``: input bra and ket on the same modes and output bra and ket on the same modes
        - ``CircuitComponent``: otherwise

        >>> from mrmustard.lab import Coherent, Attenuator, Ket, DM, Channel
        >>> state = Coherent(0, 1.0)
        >>> assert issubclass(Coherent, Ket)
        >>> assert issubclass(Attenuator, Channel)
        >>> assert isinstance(state >> Attenuator(0, 0.5), DM)
        >>> assert math.allclose(state >> state.dual, 1+0j)

        Args:
            other: The other component or (batchable) scalar to contract with.

        Returns:
            The contracted component or (batched) scalar value of the representation.

        Raises:
            ValueError: If the component wires are incompatible with the other component.
        """
        if hasattr(other, "__custom_rrshift__"):
            return other.__custom_rrshift__(self)

        if not isinstance(other, CircuitComponent):
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
            ret = self.adjoint.contract(self.contract(other))
        elif other_needs_bra or other_needs_ket:
            ret = self.contract(other.adjoint).contract(other)
        else:
            msg = f"``>>`` not supported between {self} and {other} because it's not clear if or "
            msg += "where to add missing wires. Use ``contract`` and specify all the components."
            raise ValueError(msg)
        return self._rshift_return(ret)

    def __sub__(self, other: CircuitComponent) -> CircuitComponent:
        r"""Implements the subtraction between circuit components."""
        if self.wires != other.wires:
            raise ValueError("Cannot subtract components with different wires.")
        ansatz = self.ansatz - other.ansatz
        name = self.name if self.name == other.name else ""
        return self._from_attributes(ansatz, self.wires, name)

    def __truediv__(self, other: Scalar) -> CircuitComponent:
        r"""Implements the division by a scalar for circuit components."""
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
