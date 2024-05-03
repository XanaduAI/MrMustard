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
A set of components that do not correspond to physical elements of a circuit, but can be used to
perform useful mathematical calculations.
"""

# pylint: disable=super-init-not-called, protected-access

from __future__ import annotations
from typing import Sequence

from mrmustard.physics import triples
from .circuit_components import CircuitComponent
from ..physics.representations import Bargmann


__all__ = ["TraceOut", "DsMap", "BtoQMap"]


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
        >>> expectation = ((state.dm() @ op) >> TraceOut([0, 1, 2])).representation.c

        >>> assert np.allclose(expectation, state.expectation(op))

    Args:
        modes: The modes to trace out.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__("Tr", modes_in_ket=modes, modes_in_bra=modes)

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.identity_Abc(len(self.modes)))


class DsMap(CircuitComponent):
    r"""The `s`-parametrized ``Dgate`` as a ``Channel``.

    Used internally as a ``Channel`` for transformations between representations.

    Args:
        num_modes: The number of modes of this channel.
        s: The `s` parameter of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
        s: float,
    ):
        super().__init__(
            "DsMap",
            modes_out_bra=modes,
            modes_in_bra=modes,
            modes_out_ket=modes,
            modes_in_ket=modes,
        )
        self.s = s

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.displacement_map_s_parametrized_Abc(self.s, len(self.modes)))


class BtoQMap(CircuitComponent):
    r"""The kernel for the change of representation from ``Bargmann`` into quadrature.

    Used internally as a ``Unitary`` for transformations between representations on the ``Ket`` Wire.

    The ``adjoint`` of this ``CircuitComponent`` denotes the change of representation kernel from ``Bargmann`` into quadrature on the `bra` Wire.
    The ``dual`` of this ``CircuitComponent`` denotes the change of representation kernel from quadrature into Bargmann.

    Args:
        modes: The modes of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(
            "BtoQMap",
            modes_out_ket=modes,
            modes_in_ket=modes,
        )

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.bargmann_to_quadrature_Abc(len(self.modes)))
