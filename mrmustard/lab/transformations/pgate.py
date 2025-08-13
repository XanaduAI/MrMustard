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
The class representing a quadratic phase gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ...physics import symplectics

from .base import Unitary

__all__ = ["Pgate"]


class Pgate(Unitary):
    r"""
    Quadratic phase gate.

    Args:
        mode: The mode this gate is applied to.
        shearing: The shearing parameter.

    .. details::
        The quadratic phase gate is defined as

        .. math::

            P = \exp(i s q^2 / 2 \hbar)

    Reference: https://strawberryfields.ai/photonics/conventions/gates.html
    """

    short_name = "P"

    def __init__(
        self,
        mode: int | tuple[int],
        shearing: float | Sequence[float] = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        self.shearing = shearing
        
        A, b, c = Unitary.from_symplectic(
            (mode,),
            symplectics.pgate_symplectic(1, shearing),
        ).bargmann_triple()
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket=set(mode), modes_out_ket=set(mode))
        
        super().__init__(ansatz=ansatz, wires=wires, name="Pgate")
