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

from mrmustard.physics.ansatz import ArrayAnsatz
from mrmustard.physics.fock_utils import fock_state
from mrmustard.physics.wires import ReprEnum

from ..utils import make_parameter
from .ket import Ket

__all__ = ["Number"]


class Number(Ket):
    r"""
    The number state in Fock representation.


    Args:
        mode: The mode of the number state.
        n: The number of photons.
        cutoffs: The cutoffs. If ``cutoffs`` is ``None``, it
            defaults to ``n+1``.

    .. code-block::

        >>> from mrmustard.lab import Number
        >>> from mrmustard.physics.ansatz import ArrayAnsatz

        >>> state = Number(mode=0, n=10)
        >>> assert isinstance(state.ansatz, ArrayAnsatz)

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
        mode: int,
        n: int,
        cutoff: int | None = None,
    ) -> None:
        cutoff = n if cutoff is None else cutoff
        super().__init__(name="N")
        self.parameters.add_parameter(make_parameter(False, n, "n", (None, None), dtype="int64"))
        self.parameters.add_parameter(
            make_parameter(False, cutoff, "cutoff", (None, None), dtype="int64")
        )
        self._representation = self.from_ansatz(
            modes=(mode,),
            ansatz=ArrayAnsatz.from_function(
                fock_state, n=self.parameters.n, cutoffs=self.parameters.cutoff
            ),
        ).representation
        self.short_name = str(int(n))
        self.manual_shape[0] = cutoff + 1

        for w in self.representation.wires.output.wires:
            w.repr = ReprEnum.FOCK
            w.repr_params_func = lambda w=w: [int(self.parameters.n.value)]
