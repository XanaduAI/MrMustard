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

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from typing import Optional, Sequence
from mrmustard import math
from mrmustard.lab_dev.wires import Wires
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard import physics
from mrmustard.physics.bargmann import Au2Symplectic, Symplectic2Au
from ..circuit_components import CircuitComponent

__all__ = ["Transformation", "Operation", "Unitary", "Map", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: Optional[str] = None,
    ) -> Operation:
        r"""
        Initialize an Operation from the given quadrature triple (A, b, c).
        The triple parametrizes the quadrature representation of the transformation as
        :math:`c * exp(0.5*x^T A x + b^T x)`.
        """
        from mrmustard.lab_dev.circuit_components_utils import BtoQ

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = cls(modes_out, modes_in, Bargmann(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return cls(modes_out, modes_in, BB.representation, name)

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
    ) -> Operation:
        r"""
        Initialize a Transformation from the given Bargmann triple (A,b,c)
        which parametrizes the Bargmann function of the transformation as
        :math:`c * exp(0.5*z^T A z + b^T z)`.
        """
        return cls(modes_out, modes_in, Bargmann(*triple), name)

    def inverse(self) -> Transformation:
        r"""
        Returns the mathematical inverse of the transformation, if it exists.
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
        almost_identity = self @ almost_inverse
        invert_this_c = almost_identity.representation.c
        actual_inverse = self._from_attributes(
            Bargmann(math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 / invert_this_c),
            self.wires,
            self.name + "_inv",
        )
        return actual_inverse


class Operation(Transformation):
    r"""
    A CircuitComponent with input and output wires on the ket side. Operation are allowed
    to have a different number of input and output wires.
    """

    short_name = "Op"

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
            name=name,
        )


class Unitary(Operation):
    r"""
    Base class for all unitary transformations.
    Note the default initializer is in the parent class ``Operation``.

    Arguments:
        modes_out: The output modes of this Unitary.
        modes_in: The input modes of this Unitary.
        representation: The representation of this Unitary.
        name: The name of this Unitary.
    """

    short_name = "U"

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        For example ``u >> channel`` is equivalent to ``u.adjoint @ u @ channel`` because the
        channel requires an input on the bra side as well.

        Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
        ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, Unitary):
            return Unitary._from_attributes(ret.representation, ret.wires)
        elif isinstance(other, Channel):
            return Channel._from_attributes(ret.representation, ret.wires)
        return ret

    @classmethod
    def from_symplectic(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        symplectic: tuple,
        name: Optional[str] = None,
    ) -> Unitary:
        r"""
        Initialize a Unitary from the given symplectic matrix in qqpp basis,
        i.e. the axes are ordered as [q0, q1, ..., p0, p1, ...].
        """
        M = len(modes_in) + len(modes_out)
        if symplectic.shape[-2:] != (M, M):
            raise ValueError(
                "Symplectic matrix and number of modes don't match. "
                + f"Modes imply shape {(M,M)}, "
                + f"but shape is {symplectic.shape[-2:]}."
            )
        A, b, c = physics.bargmann.wigner_to_bargmann_U(symplectic, math.zeros(M))
        return Unitary._from_attributes(
            representation=Bargmann(A, b, c),
            wires=Wires(set(), set(), set(modes_out), set(modes_in)),
            name=name,
        )

    def inverse(self) -> Unitary:
        unitary_dual = self.dual
        return Unitary._from_attributes(
            representation=unitary_dual.representation,
            wires=unitary_dual.wires,
            name=unitary_dual.name,
        )

    @property
    def symplectic(self):
        r"""
        Returns the symplectic representation
        """
        batch = self.representation.A.shape[0]
        m = self.representation.A.shape[-1]
        S = math.zeros([batch, m, m])
        for bat in range(batch):
            S[bat, :, :] = Au2Symplectic(self.representation.A[bat, :, :])
        return S

    @classmethod
    def from_symplectic_directly(cls, modes, S) -> Unitary:
        r"""
        A simple method for initializing using symplectic representation
        modes: the modes that we want the unitary to act on
        S: the symplectic representation (in XXPP order)
        """
        m = len(S)
        A = Symplectic2Au(S)
        b = [0] * m
        c = 1  # change after poly*exp ansatz
        u = Unitary.from_bargmann(modes, modes, [A, b, c])
        v = u >> u.dual
        _, _, c_prime = v.bargmann
        c = 1 / math.sqrt(c_prime[0])
        return Unitary.from_bargmann(modes, modes, [A, b, c])


class Map(Transformation):
    r"""
    A CircuitComponent more general than Channels, which are CPTP Maps.

    Arguments:
        modes_out: The output modes of this Map.
        modes_in: The input modes of this Map.
        representation: The representation of this Map.
        name: The name of this Map.
    """

    short_name = "Map"

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
            name=name or self.__class__.__name__,
        )


class Channel(Map):
    r"""
    Base class for all CPTP channels.

    Arguments:
        modes_out: The output modes of this Channel.
        modes_in: The input modes of this Channel.
        representation: The representation of this Channel.
        name: The name of this Channel
    """

    short_name = "Ch"

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Channel`` or a ``Unitary``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)
        if isinstance(other, (Channel, Unitary)):
            return Channel._from_attributes(ret.representation, ret.wires)
        return ret
