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

from abc import abstractmethod
from collections.abc import Sequence

from mrmustard import math, settings
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import XY_of_channel, au2Symplectic, symplectic2Au
from mrmustard.physics.triples import XY_to_channel_Abc
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, RealMatrix, Vector

from ..circuit_components import CircuitComponent

__all__ = ["Channel", "Map", "Operation", "Transformation", "Unitary"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    @classmethod
    @abstractmethod
    def from_ansatz(
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
        return cls.from_ansatz(modes_out, modes_in, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_fock(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        array: ComplexTensor,
        batch_dims: int = 0,
        name: str | None = None,
    ) -> Transformation:
        r"""
        Initializes a transformation of type ``cls`` given modes and a fock array.

        Args:
            modes_out: The output modes of this transformation.
            modes_in: The input modes of this transformation.
            array: The fock array of this transformation.
            batch_dims: The number of batch dimensions in the given array.
            name: The name of this transformation.

        Returns:
            A transformation in the Fock representation.
        """
        return cls.from_ansatz(modes_in, modes_out, ArrayAnsatz(array, batch_dims), name)

    @classmethod
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
        from ..circuit_components_utils.b_to_q import BtoQ  # noqa: PLC0415

        QtoB_out = BtoQ(modes_out, phi).inverse()
        QtoB_in = BtoQ(modes_in, phi).inverse().dual
        QQ = cls.from_ansatz(modes_out, modes_in, PolyExpAnsatz(*triple))
        BB = QtoB_in >> QQ >> QtoB_out
        return cls.from_ansatz(modes_out, modes_in, BB.ansatz, name)

    def inverse(self) -> Transformation:
        r"""
        Returns the mathematical inverse of the transformation, if it exists.
        Note that it can be unphysical, for example when the original is not unitary.

        Returns:
            The inverse of the transformation.

        Raises:
            NotImplementedError: If the input and output wires have different lengths.
            NotImplementedError: If the transformation is not in the Bargmann representation.

        .. code-block::
            >>> from mrmustard.lab import GDM, Identity

            >>> rho = GDM(0, beta = 0.1)
            >>> rho_as_operator = Operation.from_bargmann([0], [0], rho.ansatz.triple)
            >>> assert rho_as_operator >> rho_as_operator.inverse() == Identity([0])
        """
        if not len(self.wires.input) == len(self.wires.output):
            raise NotImplementedError(
                "Only Transformations with the same number of input and output wires are supported.",
            )
        if not isinstance(self.ansatz, PolyExpAnsatz):  # pragma: no cover
            raise NotImplementedError("Only Bargmann representation is supported.")

        A_orig, b_orig, c_orig = self.ansatz.triple
        A, b, _ = self.dual.ansatz.conj.triple
        A_inv = math.inv(A)
        b_of_inverse = math.einsum("...ij,...j->...i", -math.inv(A), b)

        in_idx = self.wires.input.indices
        out_idx = self.wires.output.indices
        A_orig_out = A_orig[..., out_idx, :][..., :, out_idx]
        A_inv_in = A_inv[..., in_idx, :][..., :, in_idx]
        b_orig_out = b_orig[..., out_idx]
        b_of_inverse_in = b_of_inverse[..., in_idx]
        m = A.shape[-1] // 2
        Im = math.broadcast_to(math.eye(m, dtype=math.complex128), (*A.shape[:-2], m, m))
        M = math.block([[A_orig_out, -Im], [-Im, A_inv_in]])
        combined_b = math.concat([b_orig_out, b_of_inverse_in], axis=-1)
        c_of_inverse = (
            1
            / c_orig
            * math.sqrt(math.cast(math.det(1j * M), dtype=math.complex128))
            * math.exp(
                0.5 * math.einsum("...i,...ij,...j->...", combined_b, math.inv(M), combined_b),
            )
        )

        return self._from_attributes(
            PolyExpAnsatz(
                math.inv(A),
                math.einsum("...ij,...j->...i", -math.inv(A), b),
                c_of_inverse,
            ),
            self.wires.copy(new_ids=True),
            self.name + "_inv",
        )


class Operation(Transformation):
    r"""
    A CircuitComponent with input and output wires on the ket side. Operation are allowed
    to have a different number of input and output wires.
    """

    short_name = "Op"

    @classmethod
    def from_ansatz(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        if not isinstance(modes_out, set) and sorted(modes_out) != list(modes_out):
            raise ValueError(f"Output modes must be sorted. got {modes_out}")
        if not isinstance(modes_in, set) and sorted(modes_in) != list(modes_in):
            raise ValueError(f"Input modes must be sorted. got {modes_in}")
        return Operation(
            ansatz=ansatz,
            wires=Wires(set(), set(), set(modes_out), set(modes_in)),
            name=name,
        )


class Unitary(Operation):
    r"""
    Base class for all unitary transformations.
    """

    short_name = "U"

    @property
    def symplectic(self):
        r"""
        Returns the symplectic matrix that corresponds to this unitary.
        """
        return au2Symplectic(self.ansatz.A)

    @classmethod
    def from_ansatz(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        if not isinstance(modes_out, set) and sorted(modes_out) != list(modes_out):
            raise ValueError(f"Output modes must be sorted. got {modes_out}")
        if not isinstance(modes_in, set) and sorted(modes_in) != list(modes_in):
            raise ValueError(f"Input modes must be sorted. got {modes_in}")
        return Unitary(
            ansatz=ansatz,
            wires=Wires(set(), set(), set(modes_out), set(modes_in)),
            name=name,
        )

    @classmethod
    def from_symplectic(cls, modes: Sequence[int], S: RealMatrix) -> Unitary:
        r"""
        A method for constructing a ``Unitary`` from its symplectic representation

        Args:
            modes: the modes that we want the unitary to act on (should be a list of int)
            S: the symplectic representation (in XXPP order)

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Unitary, Identity

            >>> S = math.eye(2)
            >>> U = Unitary.from_symplectic([0], S)

            >>> assert U == Identity([0])
        """
        m = len(modes)
        batch_shape = S.shape[:-2]
        A = symplectic2Au(S)
        b = math.zeros((*batch_shape, 2 * m), dtype="complex128")
        A_inin = A[..., m:, m:]
        c = ((-1) ** m * math.det(A_inin @ math.conj(A_inin) - math.eye_like(A_inin))) ** 0.25
        return Unitary.from_bargmann(modes, modes, (A, b, c))

    @classmethod
    def random(cls, modes: Sequence[int], max_r: float = 1.0) -> Unitary:
        r"""
        Returns a random unitary.

        Args:
            modes: The modes of the unitary.
            max_r: The maximum squeezing parameter.

        .. code-block::

            >>> from mrmustard.lab import Unitary

            >>> U = Unitary.random((0, 1, 2), max_r=1.2)
            >>> assert U.modes == (0,1,2)
        """
        m = len(modes)
        S = math.random_symplectic(m, max_r)
        return Unitary.from_symplectic(modes, S)

    def inverse(self) -> Unitary:
        r"""
        Returns the inverse of the unitary.

        .. code-block::

            >>> from mrmustard.lab import Unitary, Identity

            >>> u = Unitary.random((0, 1, 2))
            >>> assert u >> u.inverse() == Identity(u.modes)
        """
        unitary_dual = self.dual
        return Unitary(
            ansatz=unitary_dual.ansatz,
            wires=unitary_dual.wires,
            name=unitary_dual.name,
        )

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns:
            Contraction of ``self`` and ``other``.

        .. details::
            For example ``u >> channel`` is equivalent to ``u.adjoint @ u @ channel`` because the
            channel requires an input on the bra side as well.

            Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
            ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, Unitary):
            return Unitary(ret.ansatz, ret.wires)
        if isinstance(other, Channel):
            return Channel(ret.ansatz, ret.wires)
        return ret


class Map(Transformation):
    r"""
    A ``CircuitComponent`` more general than ``Channel``, which are CPTP maps.
    """

    short_name = "Map"

    @classmethod
    def from_ansatz(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        if not isinstance(modes_out, set) and sorted(modes_out) != list(modes_out):
            raise ValueError(f"Output modes must be sorted. got {modes_out}")
        if not isinstance(modes_in, set) and sorted(modes_in) != list(modes_in):
            raise ValueError(f"Input modes must be sorted. got {modes_in}")
        return Map(
            ansatz=ansatz,
            wires=Wires(set(modes_out), set(modes_in), set(modes_out), set(modes_in)),
            name=name,
        )


class Channel(Map):
    r"""
    Base class for all CPTP channels.
    """

    short_name = "Ch"

    @property
    def is_CP(self) -> bool:
        r"""
        Whether this channel is completely positive (CP).

        .. code-block::

            >>> from mrmustard.lab import Channel

            >>> channel = Channel.random((0, 1, 2))
            >>> assert channel.is_CP
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError(
                "Physicality conditions are not implemented for a mixture of states.",
            )
        if self.ansatz.num_derived_vars > 0:
            raise NotImplementedError(
                "Physicality conditions are not implemented for derived variables.",
            )
        if isinstance(self.ansatz, ArrayAnsatz):
            raise NotImplementedError(
                "Physicality conditions are not implemented for states with ArrayAnsatz.",
            )
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[..., :m, m:]

        if (
            math.real(math.norm(gamma_A - math.conj(math.einsum("...ij->...ji", gamma_A))))
            > settings.ATOL
        ):  # checks if gamma_A is Hermitian
            return False

        return math.all(math.real(math.eigvals(gamma_A)) > -settings.ATOL)

    @property
    def is_TP(self) -> bool:
        r"""
        Whether this channel is trace preserving (TP).

        .. code-block::

            >>> from mrmustard.lab import Channel

            >>> channel = Channel.random((0, 1, 2))
            >>> assert channel.is_TP
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError(
                "Physicality conditions are not implemented for a mixture of states.",
            )
        if self.ansatz.num_derived_vars > 0:
            raise NotImplementedError(
                "Physicality conditions are not implemented for derived variables.",
            )
        if isinstance(self.ansatz, ArrayAnsatz):
            raise NotImplementedError(
                "Physicality conditions are not implemented for states with ArrayAnsatz.",
            )
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[..., :m, m:]
        lambda_A = A[..., m:, m:]
        temp_A = (
            gamma_A
            + math.conj(math.einsum("...ij->...ji", lambda_A))
            @ math.inv(math.eye(m) - math.einsum("...ij->...ji", gamma_A))
            @ lambda_A
        )
        return math.all(math.real(math.norm(temp_A - math.eye(m))) < settings.ATOL)

    @property
    def is_physical(self) -> bool:
        r"""
        Whether this channel is physical (i.e. CPTP).

        .. code-block::

            >>> from mrmustard.lab import Channel

            >>> channel = Channel.random((0, 1, 2))
            >>> assert channel.is_physical
        """
        return self.is_CP and self.is_TP

    @property
    def XY(self) -> tuple[ComplexMatrix, ComplexMatrix]:
        r"""
        Returns the X and Y matrix corresponding to the channel.

        .. code-block::

            >>> from mrmustard.lab import Channel

            >>> channel = Channel.random((0, 1))
            >>> X, Y = channel.XY
            >>> assert X.shape == (4, 4)
            >>> assert Y.shape == (4, 4)
        """
        return XY_of_channel(self.ansatz.A)

    @classmethod
    def from_ansatz(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Transformation:
        if not isinstance(modes_out, set) and sorted(modes_out) != list(modes_out):
            raise ValueError(f"Output modes must be sorted. got {modes_out}")
        if not isinstance(modes_in, set) and sorted(modes_in) != list(modes_in):
            raise ValueError(f"Input modes must be sorted. got {modes_in}")
        return Channel(
            ansatz=ansatz,
            wires=Wires(set(modes_out), set(modes_in), set(modes_out), set(modes_in)),
            name=name,
        )

    @classmethod
    def from_XY(
        cls,
        modes_out: Sequence[int],
        modes_in: Sequence[int],
        X: RealMatrix,
        Y: RealMatrix,
        d: Vector | None = None,
    ) -> Channel:
        r"""
        Initialize a Channel from its XY representation.

        Args:
            modes: The modes the channel is defined on.
            X: The X matrix of the channel.
            Y: The Y matrix of the channel.
            d:  The d vector of the channel.


        .. code-block::

            >>> from mrmustard.lab import Attenuator, Channel

            >>> X = math.eye(2)
            >>> Y = math.zeros((2,2))
            >>> channel = Channel.from_XY([0], [0], X,Y)

            >>> assert channel == Attenuator(0, transmissivity=1)

        Raises:
            ValueError: If the dimensions of the X,Y matrices and the number of modes don't match.

        .. details::

            Each Gaussian channel transforms a state with covarince matrix :math:`\Sigma` and mean :math:`\mu`
            into a state with covariance matrix :math:`X \Sigma X^T + Y` and vector of means :math:`X\mu + d`.
            This channel has a Bargmann triple that is computed in https://arxiv.org/pdf/2209.06069. We borrow
            the formulas from the paper to implement the corresponding channel.
        """

        if X.shape[-2:] != (2 * len(modes_out), 2 * len(modes_in)) or Y.shape[-2:] != (
            2 * len(modes_out),
            2 * len(modes_out),
        ):
            raise ValueError(
                f"The dimension of XY matrices ({X.shape}, {Y.shape}) and number of modes ({len(modes_in), len(modes_out)}) don't match.",
            )

        return Channel.from_bargmann(modes_out, modes_in, XY_to_channel_Abc(X, Y, d))

    @classmethod
    def random(cls, modes: Sequence[int], max_r: float = 1.0) -> Channel:
        r"""
        A random channel without displacement.

        Args:
            modes: The modes of the channel.
            max_r: The maximum squeezing parameter.

        .. code-block::

            >>> from mrmustard.lab import Channel

            >>> channel = Channel.random((0, 1, 2), max_r=1.2)
            >>> assert channel.modes == (0, 1, 2)
        """
        from mrmustard.lab.states import Vacuum  # noqa: PLC0415

        m = len(modes)
        U = Unitary.random(range(3 * m), max_r)
        u_psi = Vacuum(range(2 * m)) >> U
        ansatz = u_psi.ansatz
        kraus = ansatz.conj.contract(
            ansatz,
            idx1=list(range(4 * m)),
            idx2=list(range(2 * m)) + list(range(4 * m, 6 * m)),
            idx_out=list(range(2 * m, 6 * m)),
        )
        return Channel.from_bargmann(modes, modes, kraus.triple)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Channel`` or a ``Unitary``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)
        if isinstance(other, Channel | Unitary):
            return Channel(ret.ansatz, ret.wires)
        return ret
