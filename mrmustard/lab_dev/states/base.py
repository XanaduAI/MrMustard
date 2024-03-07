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
This module contains the base classes for the available quantum states.

In the docstrings defining the available states we provide a definition in terms of
the covariance matrix :math:`V` and the vector of means :math:`r`. Additionally, we
provide the ``(A, b, c)`` triples that define the states in the Fock Bargmann
representation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..circuit_components import CircuitComponent
from ..transformations.transformations import Unitary, Channel

__all__ = ["State", "DM", "Ket"]


class State(CircuitComponent):
    r"""
    Base class for all states.
    """

    def __lshift__(self, other: State):
        r"""
        Projects this state onto another state by using self's ``>>`` on ``other.dual``.
        """
        return self >> other.dual


class DM(State):
    r"""
    Base class for density matrices.

    Args:
        name: The name of this state.
        modes: The modes of this state.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(name, modes_out_bra=modes, modes_out_ket=modes)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``DM`` when ``other`` is a ``Unitary`` or a ``Channel``, and ``other`` acts on
        ``self``'s modes. Otherwise, it returns a ``CircuitComponent``.
        """
        component = super().__rshift__(other)

        if isinstance(other, (Unitary, Channel)) and set(other.modes).issubset(self.modes):
            dm = DM()
            dm._wires = component.wires
            dm._representation = component.representation
            return dm
        return component

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "DM")


class Ket(State):
    r"""
    Base class for all pure states, potentially unnormalized.

    Arguments:
        name: The name of this state.
        modes: The modes of this states.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(name, modes_out_ket=modes)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``State`` (either ``Ket`` or ``DM``) when ``other`` is a ``Unitary`` or a
        ``Channel``, and ``other`` acts on ``self``'s modes. Otherwise, it returns a
        ``CircuitComponent``.
        """
        component = super().__rshift__(other)

        if isinstance(other, Unitary) and set(other.modes).issubset(set(self.modes)):
            ket = Ket()
            ket._wires = component.wires
            ket._representation = component.representation
            return ket
        elif isinstance(other, Channel) and set(other.modes).issubset(set(self.modes)):
            dm = DM()
            dm._wires = component.wires
            dm._representation = component.representation
            return dm
        return component

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Ket")
