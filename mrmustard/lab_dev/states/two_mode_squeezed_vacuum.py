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

from typing import Sequence

from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples
from .base import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["TwoModeSqueezedVacuum"]


class TwoModeSqueezedVacuum(Ket):
    r"""
    The two-mode squeezed vacuum state.

    If ``r`` and/or ``phi`` are ``Sequence``\s, their length must be equal to `1`.

    .. code-block::

        >>> from mrmustard.lab_dev import TwoModeSqueezedVacuum, S2gate, Vacuum

        >>> state = TwoModeSqueezedVacuum(modes=[0, 1], r=0.3, phi=0.2)
        >>> assert state == Vacuum([0, 1]) >> S2gate([0, 1], r=0.3, phi=0.2)


    Args:
        modes: The modes of the coherent state.
        r: The squeezing magnitude.
        phi: The squeezing angles.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.
    """

    def __init__(
        self,
        modes: Sequence[int],
        r: float = 0.0,
        phi: float = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(modes=modes, name="TwoModeSqueezedVacuum")
        rs, phis = list(reshape_params(int(len(modes) / 2), r=r, phi=phi))
        self._add_parameter(make_parameter(r_trainable, rs, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.two_mode_squeezed_vacuum_state_Abc, r=self.r, phi=self.phi
        )
