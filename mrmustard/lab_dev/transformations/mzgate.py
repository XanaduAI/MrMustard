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
The class representing a Mach-Zehnder gate.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz

from .base import Unitary
from ..utils import make_parameter

__all__ = ["MZgate"]


class MZgate(Unitary):
    r"""Mach-Zehnder gate.

    It supports two conventions:
        1. if ``internal=True``, both phases act inside the interferometer: ``phi_a`` on the upper arm, ``phi_b`` on the lower arm;
        2. if ``internal = False``, both phases act on the upper arm: ``phi_a`` before the first BS, ``phi_b`` after the first BS.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        modes (optional, List[int]): the list of modes this gate is applied to
        phi_a (float): the phase in the upper arm of the MZ interferometer
        phi_a_bounds (float, float): bounds for phi_a
        phi_a_trainable (bool): whether phi_a is a trainable variable
        phi_b (float): the phase in the lower arm or external of the MZ interferometer
        phi_b_bounds (float, float): bounds for phi_b
        phi_b_trainable (bool): whether phi_b is a trainable variable
        internal (bool): whether phases are both in the internal arms (default is False)
    """

    short_name = "MZ"

    def __init__(
        self,
        modes: Sequence[int],
        phi_a: float = 0.0,
        phi_b: float = 0.0,
        phi_a_trainable: bool = False,
        phi_b_trainable: bool = False,
        phi_a_bounds: tuple[float | None, float | None] = (None, None),
        phi_b_bounds: tuple[float | None, float | None] = (None, None),
        internal: bool = False,
    ):
        super().__init__(name="MZgate")
        self.parameters.add_parameter(make_parameter(phi_a_trainable, phi_a, "phi_a", phi_a_bounds))
        self.parameters.add_parameter(make_parameter(phi_b_trainable, phi_b, "phi_b", phi_b_bounds))

        ca = math.cos(complex(phi_a))
        sa = math.sin(complex(phi_a))
        cb = math.cos(complex(phi_b))
        sb = math.sin(complex(phi_b))
        cp = math.cos(complex(phi_a + phi_b))
        sp = math.sin(complex(phi_a + phi_b))

        if internal:
            symplectic = 0.5 * math.astensor(
                [
                    [ca - cb, -sa - sb, sb - sa, -ca - cb],
                    [-sa - sb, cb - ca, -ca - cb, sa - sb],
                    [sa - sb, ca + cb, ca - cb, -sa - sb],
                    [ca + cb, sb - sa, -sa - sb, cb - ca],
                ]
            )

        else:
            symplectic = 0.5 * math.astensor(
                [
                    [cp - ca, -sb, sa - sp, -1 - cb],
                    [-sa - sp, 1 - cb, -ca - cp, sb],
                    [sp - sa, 1 + cb, cp - ca, -sb],
                    [cp + ca, -sb, -sa - sp, 1 - cb],
                ]
            )

        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda sym: Unitary.from_symplectic(modes, sym).bargmann_triple(), sym=symplectic
            ),
        ).representation
