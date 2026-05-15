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

from collections.abc import Sequence

from mrmustard import math
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import number_state
from .ket import Ket

__all__ = ["Number"]


class Number(Ket):
    r"""
    The number state in Fock representation.

    >>> from mrmustard.lab import Number
    >>> from mrmustard.physics.ansatz import ArrayAnsatz
    >>> state = Number(mode=0, n=10)
    >>> assert isinstance(state.ansatz, ArrayAnsatz)

    Args:
        mode: The mode of the number state.
        n: The (batchable) number of photons.
        cutoff: The photon cutoff. If ``cutoff`` is ``None``, it
            defaults to ``math.max(n)``.

    .. details::

        For any :math:`\bar{n} = (n_1,\:\ldots,\:n_N)`, the :math:`N`-mode number state is defined
        by

        .. math::
            \ket{\bar{n}} = \ket{n_1}\otimes\ldots\otimes\ket{n_N}\:,

        where :math:`\ket{n_j}` is the eigenstate of the number operator on mode `j` with eigenvalue
        :math:`n_j`.

    """

    short_name = "N"

    def __init__(
        self,
        mode: int | tuple[int],
        n: int | Sequence[int],
        cutoff: int | None = None,
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.FOCK: (number_state, ("n", "shape"))}
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["n"] = Parameter.from_cc_init(n, "int64", f"{self.name}/n")
        if cutoff is None:
            try:
                cutoff = int(n)
            except TypeError:
                cutoff = int(math.max(self.parameters.n.value))
        elif (max_n := math.max(self.parameters.n.value)) > cutoff:
            raise ValueError(
                f"Photon numbers cannot be larger than the cutoff. Got max(n) = {max_n} and cutoff = {cutoff}."
            )

        self.short_name = str(n)

        for w in self.wires.output:
            w.repr = ReprEnum.FOCK
            w.fock_shape = cutoff + 1
