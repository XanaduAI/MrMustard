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
The class representing a controlled-phase gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ...physics import symplectics
from ..utils import make_parameter
from .base import Unitary

__all__ = ["CZgate"]


class CZgate(Unitary):
    r"""
    Controlled Z gate.

    Args:
        modes: The pair of modes of the controlled-Z gate.
        s: The control parameter.
        s_trainable: Whether ``s`` is trainable.
        s_bounds: The bounds for ``s``.

    .. code-block::

        >>> from mrmustard.lab import CZgate
        >>> gate = CZgate((0, 1), s=0.5)
        >>> assert gate.modes == (0, 1)
        >>> assert gate.parameters.s.value == 0.5

    .. details::
        We have that the controlled-Z gate is defined as

        .. math::

            C_Z = \exp(is q_1 \otimes q_2 / \hbar).

        Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 8.
        https://arxiv.org/pdf/1110.3234.pdf, Equation 161.
    """

    short_name = "CZ"

    def __init__(
        self,
        modes: tuple[int, int],
        s: float | Sequence[float] = 0.0,
        s_trainable: bool = False,
        s_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="CZgate")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=s_trainable, value=s, name="s", bounds=s_bounds, dtype=math.float64
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=lambda s: Unitary.from_symplectic(
                modes,
                symplectics.czgate_symplectic(s),
            ).bargmann_triple(),
            s=self.parameters.s,
        )
        self._wires = Wires(
            modes_in_bra=set(),
            modes_out_bra=set(),
            modes_in_ket=set(modes),
            modes_out_ket=set(modes),
        )
