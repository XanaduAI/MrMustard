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
The class representing a generic gaussian gate.
"""

from __future__ import annotations

from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import RealMatrix

from ...physics.ansatz import PolyExpAnsatz
from .base import Unitary

__all__ = ["Ggate"]


class Ggate(Unitary):
    r"""
    The generic N-mode Gaussian gate.

    Args:
        modes: The modes this gate is applied to.
        symplectic: The symplectic matrix of the gate in the XXPP ordering.
        Use ``math.random_symplectic(len(modes))`` to generate a random symplectic matrix if needed.

    .. code-block::

        >>> from mrmustard import math
        >>> from mrmustard.lab import Ggate, Vacuum, Identity, Ket

        >>> U = Ggate(modes=0, symplectic=math.random_symplectic(1))
        >>> assert isinstance(Vacuum(0) >> U, Ket)
        >>> assert U >> U.dual == Identity(0)
    """

    short_name = "G"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        symplectic: RealMatrix,
    ):
        modes = (modes,) if isinstance(modes, int) else tuple(modes)

        A, b, c = Unitary.from_symplectic(
            modes, symplectic
        ).bargmann_triple()  # TODO: add ggate to physics.triples
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))

        super().__init__(ansatz=ansatz, wires=wires, name="Ggate")
