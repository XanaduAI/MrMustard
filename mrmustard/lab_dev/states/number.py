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
The class representing a number state.
"""

from __future__ import annotations

from typing import Sequence

from mrmustard.physics.representations import Fock
from mrmustard.physics.fock import fock_state
from .base import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["Number"]


class Number(Ket):
    r"""
    The `N`-mode number state.

    .. code-block::

        >>> from mrmustard.lab_dev import Number

        >>> state = Number(modes=[0, 1], n=[10, 20])
        >>> assert state.representation.__class__.__name__ == "Fock"

    Args:
        modes: The modes of the number state.
        n: The number of photons in each mode.
        cutoffs: The cutoffs for the various modes. If ``cutoffs`` is given as
            an ``int``, it is broadcasted to all the states. If ``None``, it
            defaults to ``[n1+1, n2+1, ...]``, where ``ni`` is the photon number
            of the ``i``th mode.

    .. details::

        For any :math:`\bar{n} = (n_1,\:\ldots,\:n_N)`, the :math:`N`-mode number state is defined
        by

        .. math::
            \ket{\bar{n}} = \ket{n_1}\otimes\ldots\otimes\ket{n_N}\:,

        where :math:`\ket{n_j}` is the eigenstate of the number operator on mode `j` with eigenvalue
        :math:`n_j`.

    """

    def __init__(
        self,
        modes: Sequence[int],
        n: int | Sequence[int],
        cutoffs: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(modes=modes, name="N")
        ns, cs = list(reshape_params(len(modes), n=n, cutoffs=n if cutoffs is None else cutoffs))
        self._add_parameter(make_parameter(False, ns, "n", (None, None), dtype="int64"))
        self._add_parameter(make_parameter(False, cs, "cutoffs", (None, None)))
        self.short_name = [str(int(n)) for n in self.n.value]
        for i, cutoff in enumerate(self.cutoffs.value):
            self.manual_shape[i] = int(cutoff) + 1
        self._representation = Fock.from_function(
            fock_state, n=self.n.value, cutoffs=self.cutoffs.value
        )
