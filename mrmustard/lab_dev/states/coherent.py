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
The class representing a coherent state.
"""

from __future__ import annotations
from typing import Sequence
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples
from .ket import Ket
from ..utils import make_parameter

__all__ = ["Coherent"]


class Coherent(Ket):
    r"""
    The coherent state in Bargmann representation.

    .. code-block::

        >>> from mrmustard.lab_dev import Coherent, Vacuum, Dgate

        >>> state = Coherent(mode=0, x=0.3, y=0.2)
        >>> assert state == Vacuum(0) >> Dgate(0, x=0.3, y=0.2)

    Args:
        mode: The mode of the coherent state.
        x: The `x` displacement of the coherent state.
        y: The `y` displacement of the coherent state.
        x_trainable: Whether the `x` displacement is trainable.
        y_trainable: Whether the `y` displacement is trainable.
        x_bounds: The bounds of the `x` displacement.
        y_bounds: The bounds of the `y` displacement.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        coherent state displaced :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \sqrt{2\hbar}[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b=\bar{\alpha}\text{, and }c=\text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    short_name = "Coh"

    def __init__(
        self,
        mode: int,
        x: float | Sequence[float] = 0.0,
        y: float | Sequence[float] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        y_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="Coherent")
        self.parameters.add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self.parameters.add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

        self._representation = self.from_ansatz(
            modes=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.coherent_state_Abc, x=self.parameters.x, y=self.parameters.y
            ),
        ).representation
