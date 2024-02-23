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

from ..circuit_components import CircuitComponent

__all__ = ["Transformation", "Unitary", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """


class Unitary(Transformation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(name, modes_in_ket=modes, modes_out_ket=modes)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Unitary`` when ``other`` is a ``Unitary``, a ``Channel`` when ``other`` is a
        ``Channel``, and a ``CircuitComponent`` otherwise.
        """
        component = super().__rshift__(other)

        if isinstance(other, Unitary):
            unitary = Unitary()
            unitary._wires = component.wires
            unitary._representation = component.representation
            return unitary
        elif isinstance(other, Channel):
            channel = Channel()
            channel._wires = component.wires
            channel._representation = component.representation
            return channel
        return component


class Channel(Transformation):
    r"""
    Base class for all non-unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(
            name, modes_in_ket=modes, modes_out_ket=modes, modes_in_bra=modes, modes_out_bra=modes
        )

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``Channel`` when ``other`` is a ``Unitary`` or a ``Channel``, and a
        ``CircuitComponent`` otherwise.
        """
        component = super().__rshift__(other)

        if isinstance(other, (Unitary, Channel)):
            channel = Channel()
            channel._wires = component.wires
            channel._representation = component.representation
            return channel
        return component
