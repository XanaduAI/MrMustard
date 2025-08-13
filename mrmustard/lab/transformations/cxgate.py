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
The class representing a controlled-X gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ...physics import symplectics

from .base import Unitary

__all__ = ["CXgate"]


class CXgate(Unitary):
    r"""
    Controlled X gate.

    Args:
        modes: The pair of modes of the controlled-X gate.
        s: The control parameter.

    .. code-block::

        >>> from mrmustard.lab import CXgate
        >>> gate = CXgate((0, 1), s=0.5)
        >>> assert gate.modes == (0, 1)
        >>> assert gate.parameters.s.value == 0.5

    .. details::

        We have that the controlled-X gate is defined as
            .. math::

                C_X = \exp(is q_1 \otimes p_2)

        Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 9.
    """

    short_name = "CX"

    def __init__(
        self,
        modes: tuple[int, int],
        s: float | Sequence[float] = 0.0,
    ):
        self.s = s

        A, b, c = Unitary.from_symplectic(
            modes,
            symplectics.cxgate_symplectic(s),
        ).bargmann_triple()
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(
            modes_in_bra=set(),
            modes_out_bra=set(),
            modes_in_ket=set(modes),
            modes_out_ket=set(modes),
        )
        
        super().__init__(ansatz=ansatz, wires=wires, name="CXgate")
