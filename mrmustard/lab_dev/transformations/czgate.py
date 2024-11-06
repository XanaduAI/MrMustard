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
The class representing a controlled-phase gate.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz

from .base import Unitary
from ..utils import make_parameter

__all__ = ["CZgate"]


class CZgate(Unitary):
    r"""Controlled Z gate.

    It applies to a single pair of modes. One can optionally set bounds for each parameter, which
    the optimizer will respect.

    .. math::

        C_Z = \exp(ig q_1 \otimes q_2 / \hbar).


    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 8.
    https://arxiv.org/pdf/1110.3234.pdf, Equation 161.


    Args:
        modes (optional, Sequence[int]): the list of modes this gate is applied to
        s (float): control parameter
        s_bounds (float, float): bounds for the control angle
        s_trainable (bool): whether s is a trainable variable
    """

    short_name = "CZ"

    def __init__(
        self,
        modes: Sequence[int],
        s: float | None = 0.0,
        s_trainable: bool = False,
        s_bounds: tuple[float | None, float | None] = (None, None),
    ):
        if len(modes) != 2:
            raise ValueError(
                f"The number of modes for a CZgate must be 2 (your input has {len(modes)} many modes)."
            )
        super().__init__(name="CZgate")
        self._add_parameter(make_parameter(s_trainable, s, "s", s_bounds))
        symplectic = math.astensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, s, 1, 0],
                [s, 0, 0, 1],
            ]
        )
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda sym: Unitary.from_symplectic(modes, sym).bargmann_triple(), sym=symplectic
            ),
        ).representation
