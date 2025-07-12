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
The class representing a two-mode squeezing gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.wires import Wires

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from ..utils import make_parameter
from .base import Unitary

__all__ = ["S2gate"]


class S2gate(Unitary):
    r"""
    The two-mode squeezing gate.


    Args:
        modes: The pair of modes of the two-mode squeezing gate.
        r: The squeezing amplitude.
        phi: The phase angle.
        r_trainable: Whether ``r`` is trainable.
        phi_trainable: Whether ``phi`` is trainable.
        r_bounds: The bounds for ``r``.
        phi_bounds: The bounds for ``phi``.

    .. code-block::

        >>> from mrmustard.lab import S2gate

        >>> unitary = S2gate(modes=(1, 2), r=1)
        >>> assert unitary.modes == (1, 2)
        >>> assert unitary.parameters.r.value == 1
        >>> assert unitary.parameters.phi.value == 0.0

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = \begin{bmatrix}
                    O & e^{i\phi}\tanh(r) & \sech(r) & 0 \\
                    e^{i\phi}\tanh(r) & 0 & 0 & \sech(r) \\
                    \sech(r) & & 0 & 0 -e^{i\phi}\tanh(r) \\
                    O & \sech(r) & -e^{i\phi}\tanh(r) & 0
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = \sech(r)
    """

    def __init__(
        self,
        modes: tuple[int, int],
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: tuple[float | None, float | None] = (0, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="S2gate")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=r_trainable, value=r, name="r", bounds=r_bounds, dtype=math.float64
            ),
        )
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=phi_trainable,
                value=phi,
                name="phi",
                bounds=phi_bounds,
                dtype=math.float64,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.twomode_squeezing_gate_Abc,
            r=self.parameters.r,
            phi=self.parameters.phi,
        )
        self._wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))
