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

from ..utils import make_parameter
from .ket import Ket

__all__ = ["DisplacedSqueezed"]


class DisplacedSqueezed(Ket):
    r"""
    The displaced squeezed state in Bargmann representation.


    Args:
        mode: The mode of the displaced squeezed state.
        x: The displacement along the `x` axis, which represents the position axis in phase space.
        y: The displacement along the `y` axis, which represents the momentum axis in phase space.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        x_trainable: Whether `x` is a trainable variable.
        y_trainable: Whether `y` is a trainable variable.
        r_trainable: Whether `r` is a trainable variable.
        phi_trainable: Whether `phi` is a trainable variable.
        x_bounds: The bounds of `x`.
        y_bounds: The bounds of `y`.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.


    Returns:
        A ``Ket``.

    .. code-block::

        >>> from mrmustard.lab import DisplacedSqueezed, Vacuum, Sgate, Dgate

        >>> state = DisplacedSqueezed(mode=0, x=1, phi=0.2)
        >>> assert state == Vacuum(0) >> Sgate(0, phi=0.2) >> Dgate(0, x=1)
    """

    short_name = "DSq"

    def __init__(
        self,
        mode: int | tuple[int],
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
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="DisplacedSqueezed")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=x_trainable, value=x, name="x", bounds=x_bounds, dtype=math.float64
            ),
        )
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=y_trainable, value=y, name="y", bounds=y_bounds, dtype=math.float64
            ),
        )
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
            fn=triples.displaced_squeezed_vacuum_state_Abc,
            x=self.parameters.x,
            y=self.parameters.y,
            r=self.parameters.r,
            phi=self.parameters.phi,
        )
        self._wires = Wires(modes_out_ket=set(mode))
