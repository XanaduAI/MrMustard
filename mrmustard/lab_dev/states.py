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
The classes representing states in quantum circuits.
"""

from __future__ import annotations

from typing import Sequence

from mrmustard import math
from ..physics.representations import Bargmann
from ..utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Mode
from .circuits import Circuit
from .circuit_components import CircuitComponent

__all__ = ["Pure", "State", "Vacuum"]


class State(CircuitComponent):
    r"""
    Base class for all states.
    """
    def __init__(name, modes

    def __rshift__(self, other: CircuitComponent):
        r"""
        Returns a ``Circuit`` with two components, light-copied from ``self`` and ``other``.
        """
        return Circuit([self, other])


class Ket(State):
    r"""
    Base class for all pure states.

    Arguments:
        name: The name of this pure state.
        modes: The modes of this pure states.
    """

    def __init__(self, name: str, modes: Sequence[Mode]):
        super().__init__(name, modes_out_ket=modes)

class DM(State):



class Vacuum(Ket):
    r"""
    The N-mode vacuum state.

    Args:
        num_modes (int): the number of modes.
    """

    def __init__(
        self,
        num_modes: int,
    ) -> None:
        super().__init__("Vacuum", modes=list(range(num_modes)))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)
        A = math.zeros(shape=(2 * num_modes, 2 * num_modes))
        B = math.zeros(shape=(2 * num_modes))
        C = 1
        return Bargmann(A, B, C)
