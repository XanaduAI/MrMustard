# Copyright 2024 Xanadu Quantum Technologies Inc.

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
The class representing a trace out operation.
"""

from __future__ import annotations
from typing import Sequence

from mrmustard import math
from mrmustard.physics import triples

from ..circuit_components import CircuitComponent
from ...physics.representations import Bargmann

__all__ = ["TraceOut"]


class TraceOut(CircuitComponent):
    r"""
    A circuit component to perform trace-out operations.

    It has input wires on both the ket and bra sides, but no output wires. Its representation is
    the same as that of the identity channel.

    .. code-block::

        >>> from mrmustard.lab_dev import *
        >>> import numpy as np

        >>> # initialize a multi-mode state
        >>> state = Coherent([0, 1, 2], x=1)

        >>> # trace out some of the modes
        >>> assert state >> TraceOut([0]) == Coherent([1, 2], x=1).dm()
        >>> assert state >> TraceOut([1, 2]) == Coherent([0], x=1).dm()

        >>> # use the trace out to estimate expectation values of operators
        >>> op = Dgate([0], x=1)
        >>> expectation = (state.dm() @ op) >> TraceOut([0, 1, 2])

        >>> assert np.allclose(expectation, state.expectation(op))

    Args:
        modes: The modes to trace out.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(
            modes_in_ket=modes,
            modes_in_bra=modes,
            representation=Bargmann.from_function(fn=triples.identity_Abc, n_modes=len(modes)),
            name="Tr",
        )

    def __custom_rrshift__(self, other: CircuitComponent | complex) -> CircuitComponent | complex:
        r"""A custom ``>>`` operator for the ``TraceOut`` component.
        It allows ``TraceOut`` to carry the method that processes ``other >> TraceOut``.
        We know that the trace in Bargmann is a Gaussian integral, and in
        Fock it's a trace (rather than an inner product with the identity).
        So we write two shortcuts here, and ``__rrshift__`` will be called first if
        present in the ``__rshift__`` method of the first object (``other`` here).
        """
        ket = other.wires.output.ket
        bra = other.wires.output.bra
        idx_zconj = [bra[m].indices[0] for m in self.wires.modes & bra.modes]
        idx_z = [ket[m].indices[0] for m in self.wires.modes & ket.modes]
        if len(self.wires) == 0:
            repr = other.representation
            wires = other.wires
        elif not ket or not bra:
            repr = other.representation.conj()[idx_z] @ other.representation[idx_z]
            wires, _ = (other.wires.adjoint @ other.wires)[0] @ self.wires
        else:
            repr = other.representation.trace(idx_z, idx_zconj)
            wires, _ = other.wires @ self.wires

        cpt = other._from_attributes(repr, wires)  # pylint:disable=protected-access
        return math.sum(cpt.representation.scalar) if len(cpt.wires) == 0 else cpt
