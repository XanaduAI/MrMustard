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

from typing import Sequence

from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples
from .base import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["DisplacedSqueezed"]


class DisplacedSqueezed(Ket):
    r"""
    The `N`-mode displaced squeezed vacuum state.

    If ``x``, ``y``, ``r``, and/or ``phi`` are ``Sequence``\s, their length must be equal to `1`
    or `N`. If their length is equal to `1`, all the modes share the same parameters.

    .. code-block::

        >>> from mrmustard.lab_dev import DisplacedSqueezed, Vacuum, Sgate, Dgate

        >>> state = DisplacedSqueezed(modes=[0, 1, 2], x=1, phi=0.2)
        >>> assert state == Vacuum([0, 1, 2]) >> Sgate([0, 1, 2], phi=0.2) >> Dgate([0, 1, 2], x=1)

    Args:
        modes: The modes of the coherent state.
        x: The displacements along the `x` axis, which represents position axis in phase space.
        y: The displacements along the `y` axis.
        r: The squeezing magnitude.
        phi: The squeezing angles.
        x_trainable: Whether `x` is a trainable variable.
        y_trainable: Whether `y` is a trainable variable.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        x_bounds: The bounds for the displacement along the `x` axis.
        y_bounds: The bounds for the displacement along the `y` axis, which represents momentum axis in phase space.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.
    """

    short_name = "DSq"

    def __init__(
        self,
        modes: Sequence[int],
        x: float | Sequence[float] = 0.0,
        y: float | Sequence[float] = 0.0,
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        y_bounds: tuple[float | None, float | None] = (None, None),
        r_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(modes=modes, name="DisplacedSqueezed")
        params = reshape_params(len(modes), x=x, y=y, r=r, phi=phi)
        xs, ys, rs, phis = list(params)
        self._add_parameter(make_parameter(x_trainable, xs, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, ys, "y", y_bounds))
        self._add_parameter(make_parameter(r_trainable, rs, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.displaced_squeezed_vacuum_state_Abc,
            x=self.x,
            y=self.y,
            r=self.r,
            phi=self.phi,
        )
