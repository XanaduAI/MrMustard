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

from typing import Iterable, Optional, Tuple, Union

from .base import Ket
from ..utils import make_parameter, reshape_params
from ...physics.representations import Bargmann, Fock
from ...physics import triples

__all__ = ["Coherent", "Number", "Vacuum"]


class Coherent(Ket):
    r"""The `N`-mode coherent state.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Coherent, Vacuum, Dgate

        >>> state = Coherent(modes=[0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)
        >>> assert state.modes == [0, 1, 2]

    Args:
        modes: The modes of the coherent state.
        x: The `x` displacement of the coherent state.
        y: The `y` displacement of the coherent state.
        x_trainable: Whether the `x` displacement is trainable.
        y_trainable: Whether the `y` displacement is trainable.
        x_bounds: The bounds of the `x` displacement.
        y_bounds: The bounds of the `y` displacement.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        coherent state displaced :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \sqrt{2\hbar}[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b=\bar{\alpha}\text{, and }c=\text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    def __init__(
        self,
        modes: Iterable[int],
        x: Union[float, Iterable[float]] = 0.0,
        y: Union[float, Iterable[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__("Coherent", modes=modes)
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        xs, ys = list(reshape_params(n_modes, x=self.x.value, y=self.y.value))
        return Bargmann(*triples.coherent_state_Abc(xs, ys))


class Number(Ket):
    r"""The `N`-mode number state.
    """
    def __init__(
        self,
        modes: Iterable[int],
        n: int | Iterable[int]
    ) -> None:
        super().__init__("Number", modes=modes)
        self._add_parameter(make_parameter(False, n, "n", (None, None)))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        ns = list(reshape_params(n_modes, x=self.x.value, y=self.y.value))
        return Bargmann(*triples.coherent_state_Abc(xs, ys))


class Vacuum(Ket):
    r"""
    The `N`-mode vacuum state.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Vacuum([1, 2])
        >>> assert state.modes == [1, 2]

    Args:
        modes: A list of modes.

    .. details::

        The :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \bar{0}_N.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b = O_N\text{, and }c = 1.
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
