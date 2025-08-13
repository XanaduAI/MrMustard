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
from mrmustard.physics.ansatz import ArrayAnsatz
from mrmustard.physics.fock_utils import fock_state
from mrmustard.physics.wires import ReprEnum, Wires

from .ket import Ket

__all__ = ["Number"]


class Number(Ket):
    r"""
    The number state in Fock representation.

    Args:
        mode: The mode of the number state.
        n: The (batchable) number of photons.
        cutoff: The photon cutoff. If ``cutoff`` is ``None``, it
            defaults to ``math.max(n)+1``.

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
        mode: int | tuple[int],
        n: int | Sequence[int],
        cutoff: int | None = None,
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        n_tensor = math.astensor(n, dtype=math.int64)
        cutoff = int(math.max(n_tensor) + 1) if cutoff is None else cutoff
        cutoff_tensor = math.astensor(cutoff, dtype=math.int64)
        
        self.n = n_tensor
        self.cutoff = cutoff_tensor
        
        batch_dims = len(n_tensor.shape)
        array = fock_state(n=n_tensor, cutoff=cutoff)
        ansatz = ArrayAnsatz(array, batch_dims=batch_dims)
        wires = Wires(modes_out_ket=set(mode))
        
        super().__init__(ansatz=ansatz, wires=wires, name="N")
        
        self.short_name = str(int(n_tensor)) if batch_dims == 0 else "N_batched"
        self.manual_shape = (int(cutoff),)

        for w in self.wires.output:
            w.repr = ReprEnum.FOCK
            w.fock_cutoff = int(cutoff)
