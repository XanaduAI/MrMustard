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

from mrmustard import math
from .base import Ket, DM
from ..utils import make_parameter
from ...physics.representations import Bargmann
from ...physics import triples

__all__ = ["Coherent", "Vacuum"]


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

        The vacuum state :math:`\ket{0}` is a Gaussian state defined by

        .. math::
            \ket{0} = \frac{1}{\sqrt[4]{\pi \hbar}}
            \int dx~e^{-x^2/(2 \hbar)}\ket{x} ~~\text{, where}~~ \a\ket{0}=0
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


class Coherent(Ket):
    r"""The `N`-mode coherent state.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Coherent, Vacuum, Dgate

        >>> state = Coherent(modes=[0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)
        >>> assert state.modes == [0, 1, 2]
        >>> # assert Coherent(x=0.5, y=0.2) == Vacuum([0]) >> Dgate(x=0.5, y=0.2)

    Args:
        modes: The modes of the coherent state.
        x: The `x` displacement of the coherent state.
        y: The `y` displacement of the coherent state.
        x_trainable: Whether the `x` displacement is trainable.
        y_trainable: Whether the `y` displacement is trainable.
        x_bounds: The bounds of the `x` displacement.
        y_bounds: The bounds of the `y` displacement.

    .. details::

        For any :math:`\alpha = x + iy\in\mathbb{C}`, the coherent state :math:`\ket{\alpha}` is a
        displaced vacuum state defined by

        .. math::
            \ket{\alpha} = D(\alpha)\ket{0}\:,

        where :math:`D(\alpha)` is a displacement gate.  The values of ``x`` and ``y`` are
        automatically rescaled by :math:`1/(2\sqrt{\hbar})`.
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
        num_modes = len(self.modes)

        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = math.astensor([xs[0] for _ in range(num_modes)])
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = math.astensor([ys[0] for _ in range(num_modes)])

        return Bargmann(*triples.coherent_state_Abc(xs, ys))
