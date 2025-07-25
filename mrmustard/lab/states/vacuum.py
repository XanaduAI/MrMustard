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
The class repesenting a vacuum state.
"""

from __future__ import annotations

from collections.abc import Collection

from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from .ket import Ket

__all__ = ["Vacuum"]


class Vacuum(Ket):
    r"""
    The `N`-mode vacuum state in Bargmann representation.

    Args:
        modes: A tuple of modes.


    .. code-block::

        >>> from mrmustard.lab import Vacuum

        >>> state = Vacuum((1, 2))
        >>> assert state.modes == (1, 2)

    .. details::

        The :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \bar{0}_N.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b = O_N\text{, and }c = 1.
    """

    short_name = "Vac"

    def __init__(
        self,
        modes: int | tuple[int, ...],
    ) -> None:
        modes = (modes,) if isinstance(modes, int) else modes
        A, b, c = triples.vacuum_state_Abc(len(modes))
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_ket=set(modes))
        super().__init__(ansatz, wires, name="Vac")

        self.manual_shape = (1,) * len(modes)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        (modes,) = aux_data
        return cls(modes)

    def _tree_flatten(self):  # pragma: no cover
        children = ()
        aux_data = (self.modes,)
        return (children, aux_data)

    def __getitem__(self, idx: int | Collection[int]) -> Vacuum:
        idx = (idx,) if isinstance(idx, int) else idx
        if not set(idx).issubset(set(self.modes)):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{idx}``.")
        return Vacuum(idx)
