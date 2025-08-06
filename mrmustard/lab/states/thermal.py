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

from collections.abc import Sequence

from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ..utils import make_parameter
from .dm import DM

__all__ = ["Thermal"]


class Thermal(DM):
    r"""
    The thermal state in Bargmann representation.


    Args:
        mode: The mode of the thermal state.
        nbar: The expected number of photons.
        nbar_trainable: Whether ``nbar`` is trainable.
        nbar_bounds: The bounds of ``nbar``.

    Returns:
        A ``DM`` type object that represents the thermal state.

    .. code-block::

        >>> from mrmustard.lab import Vacuum

        >>> state = Thermal(1, nbar=3)
        >>> assert state.modes == (1,)
    """

    short_name = "Th"

    def __init__(
        self,
        mode: int | tuple[int],
        nbar: int | Sequence[int] = 0,
        nbar_trainable: bool = False,
        nbar_bounds: tuple[float | None, float | None] = (0, None),
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Thermal")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=nbar_trainable,
                value=nbar,
                name="nbar",
                bounds=nbar_bounds,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.thermal_state_Abc,
            nbar=self.parameters.nbar,
        )
        self._wires = Wires(modes_out_bra=set(mode), modes_out_ket=set(mode))
