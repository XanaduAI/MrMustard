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
The class representing an Interferometer gate.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard import math
from mrmustard.math.parameters import update_unitary
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.utils.typing import ComplexMatrix

from .base import Unitary
from ..utils import make_parameter

__all__ = ["Interferometer"]


class Interferometer(Unitary):
    r"""N-mode interferometer.

    It corresponds to a Ggate with zero mean and a ``2N x 2N`` unitary symplectic matrix.

    Args:
        modes (optional, Sequence[int]): the list of modes this gate is applied to
        num_modes (int): the num_modes-mode interferometer
        unitary (2d array): a valid unitary matrix U. For N modes it must have shape `(N,N)`
        unitary_trainable (bool): whether unitary is a trainable variable
    """

    short_name = "I"

    def __init__(
        self,
        modes: Sequence[int],
        num_modes: int | None = None,
        unitary: ComplexMatrix | None = None,
        unitary_trainable: bool = False,
    ):
        if num_modes is not None and num_modes != len(modes):
            raise ValueError(f"Invalid number of modes: got {len(modes)}, should be {num_modes}")
        if unitary is None:
            unitary = math.random_unitary(num_modes)
        super().__init__(name="Interferometer")
        self.parameter_set.add_parameter(
            make_parameter(unitary_trainable, unitary, "unitary", (None, None), update_unitary)
        )
        symplectic = math.block(
            [
                [math.real(self.unitary.value), -math.imag(self.unitary.value)],
                [math.imag(self.unitary.value), math.real(self.unitary.value)],
            ]
        )

        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda sym: Unitary.from_symplectic(modes, sym).bargmann_triple(), sym=symplectic
            ),
        ).representation
