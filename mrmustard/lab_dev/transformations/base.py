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

from abc import abstractmethod

from typing import Sequence
from mrmustard import math, settings
from mrmustard.physics.ansatz import PolyExpAnsatz, ArrayAnsatz
from mrmustard.physics.representations import Representation
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexMatrix
from mrmustard.physics.bargmann_utils import au2Symplectic, symplectic2Au, XY_of_channel
from ..circuit_components import CircuitComponent


__all__ = ["Transformation", "Operation", "Unitary", "Map", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    @classmethod
    @abstractmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: str | None = None,
    ) -> Transformation:
        r"""
        Initialize a Transformation from the given Bargmann triple (A,b,c)
        which parametrizes the Bargmann function of the transformation as
        :math:`c * exp(0.5*z^T A z + b^T z)`.
        """

    @classmethod
    @abstractmethod
    def from_modes(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        r"""
        Initializes a transformation of type ``cls`` given modes and an ansatz.

        Args:
            modes_out: The output modes of this transformation.
            modes_in: The input modes of this transformation.
            ansatz: The ansatz of this transformation.
            name: The name of this transformation.

        Returns:
            A transformation.
        """

    @classmethod
    @abstractmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: str | None = None,
    ) -> Transformation:
        r"""
        Initialize a Transformation from the given quadrature triple (A, b, c).
        The triple parametrizes the quadrature representation of the transformation as
        :math:`c * exp(0.5*x^T A x + b^T x)`.
        """

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
        if not isinstance(self.ansatz, PolyExpAnsatz):
            raise NotImplementedError("Only Bargmann representation is supported.")
        if self.ansatz.batch_size > 1:
            raise NotImplementedError("Batched transformations are not supported.")

        # compute the inverse
        A, b, _ = self.dual.ansatz.conj.triple  # apply X(.)X
        almost_inverse = self._from_attributes(
            Representation(
                PolyExpAnsatz(math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 + 0j), self.wires
            )
        )
        almost_identity = self @ almost_inverse
        invert_this_c = almost_identity.ansatz.c
        actual_inverse = self._from_attributes(
            Representation(
                PolyExpAnsatz(math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 / invert_this_c), self.wires
            ),
            self.name + "_inv",
        )
        return actual_inverse


class Operation(Transformation):
    r"""
    A CircuitComponent with input and output wires on the ket side. Operation are allowed
    to have a different number of input and output wires.
    """

    short_name = "Op"

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: str | None = None,
    ) -> Transformation:
        return Operation.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_modes(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        modes_out = set(modes_out)
        modes_in = set(modes_in)
        return Operation(
            representation=Representation(ansatz=ansatz, wires=Wires((), (), modes_out, modes_in)),
            name=name,
        )

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: str | None = None,
    ) -> Transformation:
        from ..circuit_components_utils.b_to_q import BtoQ

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = Operation.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return Operation.from_modes(modes_out, modes_in, BB.ansatz, name)


class Unitary(Operation):
    r"""
    Base class for all unitary transformations.
    Note the default initializer is in the parent class ``Operation``.

    Arguments:
        modes_out: The output modes of this Unitary.
        modes_in: The input modes of this Unitary.
        ansatz: The ansatz of this Unitary.
        name: The name of this Unitary.
    """

    short_name = "U"

    @property
    def symplectic(self):
        r"""
        Returns the symplectic matrix that corresponds to this unitary
        """
        batch_size = self.ansatz.batch_size
        return [au2Symplectic(self.ansatz.A[batch, :, :]) for batch in range(batch_size)]

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: str | None = None,
    ) -> Transformation:
        return Unitary.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_modes(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        modes_out = set(modes_out)
        modes_in = set(modes_in)
        return Unitary(
            representation=Representation(ansatz=ansatz, wires=Wires((), (), modes_out, modes_in)),
            name=name,
        )

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: str | None = None,
    ) -> Transformation:
        from ..circuit_components_utils.b_to_q import BtoQ

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = Unitary.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return Unitary.from_modes(modes_out, modes_in, BB.ansatz, name)

    @classmethod
    def from_symplectic(cls, modes, S) -> Unitary:
        r"""
        A simple method for initializing using symplectic representation
        modes: the modes that we want the unitary to act on (should be a list of int)
        S: the symplectic representation (in XXPP order)
        """
        m = len(modes)
        A = symplectic2Au(S)
        b = math.zeros(2 * m, dtype="complex128")
        A_inin = math.atleast_2d(A[m:, m:])
        c = ((-1) ** m * math.det(A_inin @ math.conj(A_inin) - math.eye_like(A_inin))) ** 0.25
        return Unitary.from_bargmann(modes, modes, [A, b, c])

    @classmethod
    def random(cls, modes, max_r=1):
        r"""
        Returns a random unitary.
        modes: the modes the unitary acts on non-trivially
        max_r: maximum squeezing parameter as defined in math.random_symplecic
        """
        m = len(modes)
        S = math.random_symplectic(m, max_r)
        return Unitary.from_symplectic(modes, S)

    def inverse(self) -> Unitary:
        unitary_dual = self.dual
        return Unitary(
            representation=unitary_dual.representation,
            name=unitary_dual.name,
        )

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
            return Unitary(ret.representation)
        elif isinstance(other, Channel):
            return Channel(ret.representation)
        return ret


class Map(Transformation):
    r"""
    A CircuitComponent more general than Channels, which are CPTP Maps.

    Arguments:
        modes_out: The output modes of this Map.
        modes_in: The input modes of this Map.
        ansatz: The ansatz of this Map.
        name: The name of this Map.
    """

    short_name = "Map"

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: str | None = None,
    ) -> Transformation:
        return Map.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_modes(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        modes_out = set(modes_out)
        modes_in = set(modes_in)
        return Map(
            representation=Representation(
                ansatz=ansatz, wires=Wires(modes_out, modes_in, modes_out, modes_in)
            ),
            name=name,
        )

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: str | None = None,
    ) -> Transformation:
        from ..circuit_components_utils.b_to_q import BtoQ

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = Map.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return Map.from_modes(modes_out, modes_in, BB.ansatz, name)


class Channel(Map):
    r"""
    Base class for all CPTP channels.

    Arguments:
        modes_out: The output modes of this Channel.
        modes_in: The input modes of this Channel.
        ansatz: The ansatz of this Channel.
        name: The name of this Channel
    """

    short_name = "Ch"

    @property
    def is_CP(self) -> bool:
        r"""
        Whether this channel is completely positive (CP).
        """
        batch_dim = self.ansatz.batch_size
        if batch_dim > 1:
            raise ValueError(
                "Physicality conditions are not implemented for batch dimension larger than 1."
            )
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[0, :m, m:]

        if (
            math.real(math.norm(gamma_A - math.conj(gamma_A.T))) > settings.ATOL
        ):  # checks if gamma_A is Hermitian
            return False

        return all(math.real(math.eigvals(gamma_A)) > -settings.ATOL)

    @property
    def is_TP(self) -> bool:
        r"""
        Whether this channel is trace preserving (TP).
        """
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[0, :m, m:]
        lambda_A = A[0, m:, m:]
        temp_A = gamma_A + math.conj(lambda_A.T) @ math.inv(math.eye(m) - gamma_A.T) @ lambda_A
        return math.real(math.norm(temp_A - math.eye(m))) < settings.ATOL

    @property
    def is_physical(self) -> bool:
        r"""
        Whether this channel is physical (i.e. CPTP).
        """
        return self.is_CP and self.is_TP

    @property
    def XY(self) -> tuple[ComplexMatrix, ComplexMatrix]:
        r"""
        Returns the X and Y matrix corresponding to the channel.
        """
        return XY_of_channel(self.ansatz.A[0])

    @classmethod
    def from_bargmann(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        name: str | None = None,
    ) -> Transformation:
        return Channel.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_modes(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        modes_out = set(modes_out)
        modes_in = set(modes_in)
        return Channel(
            representation=Representation(
                ansatz=ansatz, wires=Wires(modes_out, modes_in, modes_out, modes_in)
            ),
            name=name,
        )

    @classmethod
    def from_quadrature(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        triple: tuple,
        phi: float = 0,
        name: str | None = None,
    ) -> Transformation:
        from ..circuit_components_utils.b_to_q import BtoQ

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = Channel.from_modes(modes_out, modes_in, PolyExpAnsatz(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return Channel.from_modes(modes_out, modes_in, BB.ansatz, name)

    @classmethod
    def random(cls, modes: Sequence[int], max_r: float = 1.0) -> Channel:
        r"""
        A random channel without displacement.

        Args:
            modes: The modes of the channel.
            max_r: The maximum squeezing parameter.
        """
        from mrmustard.lab_dev.states import Vacuum

        m = len(modes)
        U = Unitary.random(range(3 * m), max_r)
        u_psi = Vacuum(range(2 * m)) >> U
        A = u_psi.ansatz
        kraus = A.conj[range(2 * m)] @ A[range(2 * m)]
        return Channel.from_bargmann(modes, modes, kraus.triple)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Channel`` or a ``Unitary``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)
        if isinstance(other, (Channel, Unitary)):
            return Channel(ret.representation)
        return ret
