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
The class representing a RealInterferometer gate.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard import math
from mrmustard.math.parameters import update_orthogonal
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.utils.typing import RealMatrix

from .base import Unitary
from ..utils import make_parameter

__all__ = ["RealInterferometer"]


class RealInterferometer(Unitary):
    r"""N-mode interferometer parametrized by an NxN orthogonal matrix (or 2N x 2N block-diagonal orthogonal matrix). This interferometer does not mix q and p.
    Does not mix q's and p's.

    Args:
        orthogonal (2d array, optional): a real unitary (orthogonal) matrix. For N modes it must have shape `(N,N)`.
            If set to `None` a random real unitary (orthogonal) matrix is used.
        orthogonal_trainable (bool): whether orthogonal is a trainable variable
    """

    short_name = "RI"

    def __init__(
        self,
        modes: Sequence[int],
        num_modes: int | None = None,
        orthogonal: RealMatrix | None = None,
        orthogonal_trainable: bool = False,
    ):
        if num_modes is not None and (num_modes != len(modes)):
            raise ValueError(f"Invalid number of modes: got {len(modes)}, should be {num_modes}")
        if orthogonal is None:
            orthogonal = math.random_orthogonal(num_modes)

        super().__init__(name="RealInterferometer")
        self._add_parameter(
            make_parameter(
                orthogonal_trainable, orthogonal, "orthogonal", (None, None), update_orthogonal
            )
        )
        symplectic = math.block(
            [
                [self.orthogonal.value, -math.zeros_like(self.orthogonal.value)],
                [math.zeros_like(self.orthogonal.value), self.orthogonal.value],
            ]
        )

        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda sym: Unitary.from_symplectic(modes, sym).bargmann_triple(), sym=symplectic
            ),
        ).representation
