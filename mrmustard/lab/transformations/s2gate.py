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
from .base import Unitary

__all__ = ["S2gate"]


class S2gate(Unitary):
    r"""
    The two-mode squeezing gate.


    Args:
        modes: The pair of modes of the two-mode squeezing gate.
        r: The squeezing amplitude.
        phi: The phase angle.

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
    ):
        self.r = r
        self.phi = phi
        
        A, b, c = triples.twomode_squeezing_gate_Abc(
            r=r,
            phi=phi,
        )
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))
        
        super().__init__(ansatz=ansatz, wires=wires, name="S2gate")
