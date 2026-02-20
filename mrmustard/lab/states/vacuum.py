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

from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import vacuum_state
from .ket import Ket

__all__ = ["Vacuum"]


class Vacuum(Ket):
    r"""
    The `N`-mode vacuum state in Bargmann representation.

    >>> from mrmustard.lab import Vacuum
    >>> state = Vacuum((1, 2))
    >>> assert state.modes == (1, 2)

    Args:
        modes: A tuple of modes.

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
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (vacuum_state, ("n_modes", "lin_sup"))},
                n_modes=len(modes),
            ),
            wires=Wires(modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.manual_shape = (1,) * len(modes)

    def get_modes(self, modes: int | Collection[int]) -> Vacuum:
        keep = {modes} if isinstance(modes, int) else set(modes)
        if not keep.issubset(set(self.modes)):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{keep}``.")
        return Vacuum(keep)
