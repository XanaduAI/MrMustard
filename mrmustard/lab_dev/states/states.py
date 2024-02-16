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

from typing import Iterable

from .base import Ket, DM
from ...physics.representations import Bargmann
from ...physics import triples

__all__ = ["Vacuum"]


class Vacuum(Ket):
    r"""
    The N-mode vacuum state.

    Args:
        num_modes: the number of modes.
    """

    def __init__(
        self,
        modes: Iterable[int],
    ) -> None:
        super().__init__("Vacuum", modes=modes)

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)
        return Bargmann(*triples.vacuum_state_Abc(num_modes))
