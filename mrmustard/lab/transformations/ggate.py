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

from mrmustard import math
from mrmustard.math.parameters import update_symplectic
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import RealMatrix

from ...physics.ansatz import PolyExpAnsatz
from ..utils import make_parameter
from .base import Unitary

__all__ = ["Ggate"]


class Ggate(Unitary):
    r"""
    The generic N-mode Gaussian gate.

    Args:
        modes: The modes this gate is applied to.
        symplectic: The symplectic matrix of the gate in the XXPP ordering.
        symplectic_trainable: Whether ``symplectic`` is trainable.

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
        symplectic: RealMatrix | None = None,
        symplectic_trainable: bool = False,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(name="Ggate")

        symplectic = symplectic if symplectic is not None else math.random_symplectic(len(modes))
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=symplectic_trainable,
                value=symplectic,
                name="symplectic",
                bounds=(None, None),
                update_fn=update_symplectic,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=lambda s: Unitary.from_symplectic(modes, s).bargmann_triple(),
            s=self.parameters.symplectic,
        )
        self._wires = Wires(
            modes_in_bra=set(),
            modes_out_bra=set(),
            modes_in_ket=set(modes),
            modes_out_ket=set(modes),
        )

    @property
    def symplectic(self):
        return self.parameters.symplectic.value
