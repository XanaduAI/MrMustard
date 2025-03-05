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
The class representing a thermal state.
"""

from __future__ import annotations

from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples
from .dm import DM
from ..utils import make_parameter

__all__ = ["Thermal"]


class Thermal(DM):
    r"""
    The thermal state in Bargmann representation.

    .. code-block ::

        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Thermal(1, nbar=3)
        >>> assert state.modes == (1,)

    Args:
        mode: The mode of the thermal state.
        nbar: The expected number of photons.
        nbar_trainable: Whether ``nbar`` is trainable.
        nbar_bounds: The bounds of ``nbar``.
    """

    short_name = "Th"

    def __init__(
        self,
        mode: int,
        nbar: int = 0,
        nbar_trainable: bool = False,
        nbar_bounds: tuple[float | None, float | None] = (0, None),
    ) -> None:
        super().__init__(name="Thermal")
        self.parameters.add_parameter(make_parameter(nbar_trainable, nbar, "nbar", nbar_bounds))
        self._representation = self.from_ansatz(
            modes=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.thermal_state_Abc, nbar=self.parameters.nbar
            ),
        ).representation
