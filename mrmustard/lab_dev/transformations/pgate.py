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

from typing import Sequence
from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz

from .base import Unitary
from ..utils import make_parameter

__all__ = ["Pgate"]


class Pgate(Unitary):
    r"""Quadratic phase gate.

    If len(modes) > 1 the gate is applied in parallel to all of the modes provided. If a parameter
    is a single float, the parallel instances of the gate share that parameter. To apply
    mode-specific values use a list of floats. One can optionally set bounds for each parameter,
    which the optimizer will respect.

    .. math::

        P = \exp(i s q^2 / 2 \hbar)

    Reference: https://strawberryfields.ai/photonics/conventions/gates.html

    Args:
        modes (Sequence[int]): the list of modes this gate is applied to
        shearing (float or Squence[float]): the list of shearing parameters
        shearing_bounds (float, float): bounds for the shearing parameters
        shearing_trainable bool: whether shearing is a trainable variable
    """

    short_name = "P"

    def __init__(
        self,
        modes: Sequence[int],
        shearing: float | Sequence[float] | None = 0.0,
        shearing_trainable: bool = False,
        shearing_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="Pgate")
        self.parameter_set.add_parameter(
            make_parameter(shearing_trainable, shearing, "shearing", shearing_bounds)
        )

        symplectic = math.block(
            [
                [math.eye(len(modes)), math.zeros((len(modes), len(modes)))],
                [math.eye(len(modes)) * shearing, math.eye(len(modes))],
            ]
        )
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda s: Unitary.from_symplectic(modes, s).bargmann_triple(), s=symplectic
            ),
        ).representation
