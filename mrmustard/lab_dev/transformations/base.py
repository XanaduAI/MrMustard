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
This module contains the base classes for the available unitaries and channels on quantum states.

In the docstrings defining the available unitaries we provide a definition in terms of
the symplectic matrix :math:`S` and the real vector :math:`d`. For deterministic Gaussian channels,
we use the two matrices :math:`X` and :math:`Y` and the vector :math:`d`. Additionally, we
provide the ``(A, b, c)`` triples that define the transformation in the Fock Bargmann
representation.
"""

from __future__ import annotations

from typing import Optional, Sequence
from mrmustard.utils.typing import RealMatrix, RealVector
from mrmustard import math
from mrmustard.lab_dev.wires import Wires
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard import physics
from ..circuit_components import CircuitComponent

__all__ = ["Transformation", "Operation", "Unitary", "Map", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations. Currently provides the ability to compute the inverse
    of the transformation.
    """

    def inverse(self) -> Transformation:
        r"""Returns the mathematical inverse of the transformation, if it exists.
        Note that it can be unphysical, for example when the original is not unitary.

        Returns:
            Transformation: the inverse of the transformation.

        Raises:
            NotImplementedError: if the inverse of this transformation is not supported.
        """
        if not len(self.wires.input) == len(self.wires.output):
            raise NotImplementedError(
                "Only Transformations with the same number of input and output wires are supported."
            )
        if not isinstance(self.representation, Bargmann):
            raise NotImplementedError("Only Bargmann representation is supported.")
        if self.representation.ansatz.batch_size > 1:
            raise NotImplementedError("Batched transformations are not supported.")

        # compute the inverse
        A, b, _ = self.dual.representation.conj().triple  # apply X(.)X
        almost_inverse = self._from_attributes(
            Bargmann(math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 + 0j), self.wires
        )
        almost_identity = (
            self @ almost_inverse
        )  # TODO: this is not efficient, need to get c from formula
        invert_this_c = almost_identity.representation.c
        actual_inverse = self._from_attributes(
            Bargmann(math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 / invert_this_c),
            self.wires,
            self.name + "_inv",
        )
        return actual_inverse


class Operation(Transformation):
    r"""A CircuitComponent with input and output wires, on the ket side."""

    def __init__(
        self,
        modes_out: tuple[int, ...] = (),
        modes_in: tuple[int, ...] = (),
        representation: Optional[Bargmann | Fock] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            modes_out_ket=modes_in,
            modes_in_ket=modes_out,
            representation=representation,
            name=name or "Op",
        )
        if representation is not None:
            self._representation = representation

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Operation:
        r"""Initialize an Operation from the given Bargmann triple."""
        return cls(modes_out, modes_in, Bargmann(*triple), name)

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Operation:
        r"""Initialize an Operation from the given quadrature triple."""
        from mrmustard.lab_dev.circuit_components_utils import BtoQ  # pylint: disable=import-outside-toplevel

        wires = Wires(set(), set(), set(modes_out), set(modes_in))
        QQ = cls._from_attributes(Bargmann(*triple), wires)
        QtoB_out = BtoQ(modes_out).inverse()
        QtoB_in = BtoQ(modes_in).inverse().dual
        BB = QtoB_in @ QQ @ QtoB_out
        return cls._from_attributes(BB.representation, BB.wires, name)


class Unitary(Operation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(
        self,
        modes: tuple[int, ...] = (),
        representation: Optional[Representation] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            name=name or "U" + "".join(str(m) for m in modes),
        )
        if representation is not None:
            self._representation = representation

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
        ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, Unitary):
            return Unitary._from_attributes(ret.representation, ret.wires)
        elif isinstance(other, Channel):
            return Channel._from_attributes(ret.representation, ret.wires)
        return ret

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Unitary")

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Unitary:
        r"""Initialize a Unitary from the given Bargmann triple."""
        return Unitary(modes, Bargmann(*triple), name)

    @classmethod
    def from_quadrature(
        cls,
        modes: Sequence[int],
        data: tuple,
        name: Optional[str] = None,
        phi: float = 0,
    ) -> Unitary:
        r"""Instantiates a Unitary from the given CV quadrature data.

        For the quadrature representation it is expected a tuple ``(A, b, c)``,
        where A, b and c parametrize the wavefunction in the form
        :math:`\psi(x) = c \exp(\frac{1}{2} x^T A x + b^T x)`.

        The angle phi sets the angle of the axis with respect to which the
        quadrature is intended.
        For example phi=0 corresponds to the position quadrature.
        phi=pi/2 corresponds to the momentum quadrature.

        """
        BB = super().from_quadrature(modes, modes, data)
        return Unitary(modes, BB.representation, name)

    @classmethod
    def from_symplectic(
        cls,
        modes: Sequence[int],
        symplectic: RealMatrix,
        displacement: RealVector,
        name: Optional[str] = None,
    ) -> Unitary:
        r"""Initialize a Unitary from the given symplectic matrix in qqpp basis.
        I.e. the axes are ordered as [q0, q1, ..., p0, p1, ...].
        """
        if symplectic.shape[-2:] != (2 * len(modes), 2 * len(modes)):
            raise ValueError(
                f"Symplectic matrix has incorrect shape. Expected {(2 * len(modes), 2 * len(modes))}, "
                f"got {symplectic.shape[-2:]}."
            )
        A, b, c = physics.bargmann.wigner_to_bargmann_U(symplectic, displacement)
        return Unitary._from_attributes(
            representation=Bargmann(A, b, c),
            wires=Wires(set(), set(), set(modes), set(modes)),
            name=name,
        )


class Map(Transformation):
    r"""A CircuitComponent with input and output wires and same modes on bra and ket sides.
    More general than Channels, which need to be CPTP."""

    def __init__(
        self,
        modes_out: tuple[int, ...] = (),
        modes_in: tuple[int, ...] = (),
        representation: Optional[Bargmann | Fock] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            modes_out_bra=modes_out,
            modes_in_bra=modes_in,
            modes_out_ket=modes_out,
            modes_in_ket=modes_in,
            representation=representation,
            name=name or "Map",
        )

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Map:
        r"""Initialize a Map from the given Bargmann triple."""
        return cls(modes_out, modes_in, Bargmann(*triple), name)

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Map:
        r"""Initialize a Map from the given quadrature data."""
        BBBB = super().from_quadrature(modes_out, modes_in, modes_out, modes_in, triple)
        return cls._from_attributes(BBBB.representation, BBBB.wires, name)


class Channel(Map):
    r"""
    Base class for all non-unitary channels.

    Arguments:
        name: The name of this channel.
        modes: The modes that this channel acts on.
    """

    def __init__(
        self,
        modes: tuple[int, ...] = (),
        representation: Optional[Representation] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            representation=representation,
            name=name or "Ch" + "".join(str(m) for m in modes),
        )

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Channel:
        r"""Initialize a Channel from the given Bargmann triple."""
        return Channel(modes, Bargmann(*triple), name)

    @classmethod
    def from_quadrature(
        cls, modes: Sequence[int], triple: tuple, name: Optional[str] = None
    ) -> Channel:
        r"""Instantiates a Channel from the given CV quadrature data."""
        BBBB = super().from_quadrature(modes, modes, triple)
        return Channel._from_attributes(BBBB.representation, BBBB.wires, name)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Unitary`` or a ``Channel``, and a
        ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, (Unitary, Channel)):
            return Channel._from_attributes(
                representation=ret.representation, wires=ret.wires
            )
        return ret

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Channel")
