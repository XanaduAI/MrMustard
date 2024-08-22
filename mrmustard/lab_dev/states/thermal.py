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

from typing import Sequence

from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples
from .base import DM
from ..utils import make_parameter, reshape_params

__all__ = ["Thermal"]


class Thermal(DM):
    r"""
    The `N`-mode thermal state.

    If ``nbar`` is a ``Sequence``, its length must be equal to `1` or `N`. If its length is equal to `1`,
    all the modes share the same ``nbar``.

    .. code-block ::

        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Thermal([1, 2], nbar=3)
        >>> assert state.modes == [1, 2]

    Args:
        modes: A list of modes.
        nbar: The expected number of photons in each mode.
        nbar_trainable: Whether ``nbar`` is trainable.
        nbar_bounds: The bounds of ``nbar``.
    """

    short_name = "Th"

    def __init__(
        self,
        modes: Sequence[int],
        nbar: int | Sequence[int] = 0,
        nbar_trainable: bool = False,
        nbar_bounds: tuple[float | None, float | None] = (0, None),
    ) -> None:
        super().__init__(modes=modes, name="Thermal")
        (nbars,) = list(reshape_params(len(modes), nbar=nbar))
        self._add_parameter(make_parameter(nbar_trainable, nbars, "nbar", nbar_bounds))

        self._representation = Bargmann.from_function(fn=triples.thermal_state_Abc, nbar=self.nbar)
