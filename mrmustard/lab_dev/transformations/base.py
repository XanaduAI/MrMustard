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
from mrmustard.utils.typing import ComplexMatrix, ComplexVector
from mrmustard import math
from mrmustard.lab_dev.utils import shape_check
from mrmustard.lab_dev.wires import Wires
from mrmustard.physics.representations import Bargmann
from ..circuit_components import CircuitComponent

__all__ = ["Transformation", "Unitary", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    def inverse(self) -> Transformation:
        r"""Returns the inverse of the transformation."""
        if not isinstance(self.representation, Bargmann):
            raise NotImplementedError("Only Bargmann representation is supported.")
        if self.representation.ansatz.batch_size > 1:
            raise NotImplementedError("Batched transformations are not supported.")
        A, b, _ = self.dual.representation.conj().triple  # apply X
        almost_inverse = self.__class__.from_bargmann(
            [0], (math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 + 0j)
        )
        almost_identity = (
            self >> almost_inverse
        )  # TODO: this is not efficient, need to get c from formula
        invert_this_c = almost_identity.representation.c
        actual_inverse = self.__class__.from_bargmann(
            [0], (math.inv(A[0]), -math.inv(A[0]) @ b[0], 1 / invert_this_c)
        )
        return actual_inverse


class Unitary(Transformation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: Optional[str] = None, modes: tuple[int, ...] = ()):
        super().__init__(
            name or "U" + "".join(str(m) for m in modes),
            modes_in_ket=modes,
            modes_out_ket=modes,
        )

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
        ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, Unitary):
            return Unitary._from_attributes("", ret.representation, ret.wires)
        elif isinstance(other, Channel):
            return Channel._from_attributes("", ret.representation, ret.wires)
        return ret

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Unitary")

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
    ) -> Unitary:
        A = math.astensor(triple[0])
        b = math.astensor(triple[1])
        c = math.astensor(triple[2])
        shape_check(A, b, 2 * len(modes), "Bargmann")
        s = set(modes)
        return Unitary._from_attributes(name, Bargmann(A, b, c), Wires(set(), set(), s, s))


class Channel(Transformation):
    r"""
    Base class for all non-unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: Optional[str] = None, modes: tuple[int, ...] = ()):
        super().__init__(
            name or "Ch" + "".join(str(m) for m in modes),
            modes_in_ket=modes,
            modes_out_ket=modes,
            modes_in_bra=modes,
            modes_out_bra=modes,
        )

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Unitary`` or a ``Channel``, and a
        ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if isinstance(other, (Unitary, Channel)):
            return Channel._from_attributes("", ret.representation, ret.wires)
        return ret

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Channel")

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
    ) -> Channel:
        r"""Initialize a Channel from the given Bargmann ``(A, b, c)`` triple."""
        A = math.astensor(triple[0])
        b = math.astensor(triple[1])
        c = math.astensor(triple[2])
        shape_check(A, b, 4 * len(modes), "Bargmann")
        s = set(modes)
        return Channel._from_attributes(name, Bargmann(A, b, c), Wires(s, s, s, s))
