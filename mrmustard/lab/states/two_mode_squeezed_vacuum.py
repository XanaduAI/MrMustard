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
The class representing a two-mode squeezed vacuum state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ..utils import make_parameter
from .ket import Ket

__all__ = ["TwoModeSqueezedVacuum"]


class TwoModeSqueezedVacuum(Ket):
    r"""
    The two-mode squeezed vacuum state.


    Args:
        modes: The modes of the two-mode squeezed vacuum state.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.

    Returns:
        A ``Ket`` type object that represents the two-mode squeezed vacuum state.

    .. code-block::

        >>> from mrmustard.lab import TwoModeSqueezedVacuum, S2gate, Vacuum

        >>> state = TwoModeSqueezedVacuum(modes=(0, 1), r=0.3, phi=0.2)
        >>> assert state == Vacuum((0,1)) >> S2gate((0, 1), r=0.3, phi=0.2)

    """

    def __init__(
        self,
        modes: tuple[int, int],
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="TwoModeSqueezedVacuum")
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
            fn=triples.two_mode_squeezed_vacuum_state_Abc,
            r=self.parameters.r,
            phi=self.parameters.phi,
        )
        self._wires = Wires(modes_out_ket=set(modes))
