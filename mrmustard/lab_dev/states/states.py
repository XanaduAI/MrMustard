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

from typing import Sequence, Optional, Tuple, Union

from mrmustard import math
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.physics.fock import fock_state
from mrmustard.physics import triples
from .base import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["Coherent", "Number", "Vacuum"]


class Coherent(Ket):
    r"""The `N`-mode coherent state.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block::

        >>> from mrmustard.lab_dev import Coherent, Vacuum, Dgate

        >>> state = Coherent(modes=[0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)
        >>> assert state == Vacuum([0, 1, 2]) >> Dgate([0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)

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
        modes: tuple[int, ...],
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
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

    .. code-block::

        >>> from mrmustard.lab_dev import Number

        >>> state = Number(modes=[0, 1], n=[10, 20])
        >>> assert state.modes == (0, 1)

    Args:
        modes: The modes of the number state.
        n: The number of photons in each mode.

    .. details::

        For any :math:`\bar{n} = (n_1,\:\ldots,\:n_N)`, the :math:`N`-mode number state is defined
        by

        .. math::
            \ket{\bar{n}} = \ket{n_1}\otimes\ldots\otimes\ket{n_N}\:,

        where :math:`\ket{n_j}` is the eigenstate of the number operator on mode `j` with eigenvalue
        :math:`n_j`.

    """

    def __init__(self, modes: tuple[int, ...], n: Union[int, tuple[int, ...]]) -> None:
        super().__init__("N", modes=modes)

        self._n = math.atleast_1d(n)
        if len(self._n) == 1:
            self._n = math.tile(self._n, [len(modes)])
        if len(self._n) != len(modes):
            msg = f"Length of ``n`` must be 1 or {len(modes)}, found {len(self._n)}."
            raise ValueError(msg)

    @property
    def representation(self) -> Fock:
        return Fock(fock_state(self.n))

    @property
    def n(self):
        r"""
        The number of photons in each mode.
        """
        return self._n


class Vacuum(Ket):
    r"""
    The `N`-mode vacuum state.

    .. code-block ::

        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Vacuum([1, 2])
        >>> assert state.modes == (1, 2)

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
        modes: tuple[int, ...],
    ) -> None:
        super().__init__("Vac", modes=modes)

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)
        return Bargmann(*triples.vacuum_state_Abc(num_modes))
