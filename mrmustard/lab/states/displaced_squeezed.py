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
The class representing a displaced squeezed state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from .ket import Ket

__all__ = ["DisplacedSqueezed"]


class DisplacedSqueezed(Ket):
    r"""
    The displaced squeezed state in Bargmann representation.

    Args:
        mode: The mode of the displaced squeezed state.
        alpha: The complex displacement.
        r: The squeezing magnitude.
        phi: The squeezing angle.

    Returns:
        A ``Ket``.

    .. code-block::

        >>> from mrmustard.lab import DisplacedSqueezed, Vacuum, Sgate, Dgate

        >>> state = DisplacedSqueezed(mode=0, alpha=1, r=0.2, phi=0.3)
        >>> assert state == Vacuum(0) >> Sgate(0, r=0.2, phi=0.3) >> Dgate(0, alpha=1)
    """

    short_name = "DSq"

    def __init__(
        self,
        mode: int,
        alpha: complex | Sequence[complex] = 0.0j,
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        self.alpha = alpha
        self.r = r
        self.phi = phi

        A, b, c = triples.displaced_squeezed_vacuum_state_Abc(
            alpha=alpha,
            r=r,
            phi=phi,
        )
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_ket=set(mode))
        
        super().__init__(ansatz=ansatz, wires=wires, name="DisplacedSqueezed")
