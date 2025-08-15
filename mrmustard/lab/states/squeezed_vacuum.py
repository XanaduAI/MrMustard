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
The class representing a squeezed vacuum state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexTensor

from .ket import Ket

__all__ = ["SqueezedVacuum"]


class SqueezedVacuum(Ket):
    r"""
    The squeezed vacuum state in Bargmann representation.


    Args:
        mode: The mode of the squeezed vacuum state.
        r: The squeezing magnitude.
        phi: The squeezing angle.

    .. code-block::

        >>> from mrmustard.lab import SqueezedVacuum, Vacuum, Sgate

        >>> state = SqueezedVacuum(mode=0, r=0.3, phi=0.2)
        >>> assert state == Vacuum(0) >> Sgate(0, r=0.3, phi=0.2)
    """

    short_name = "Sq"

    def __init__(
        self,
        mode: int,
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
    ):
        A, b, c = triples.squeezed_vacuum_state_Abc(
            r=r,
            phi=phi,
        )
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_ket={mode})

        def specialized_fock(shape, **kwargs):
            """Optimized Fock computation using squeezed vacuum formula."""
            r_tensor = math.astensor(r)
            phi_tensor = math.astensor(phi)
            if ansatz.batch_shape:
                rs, phis = math.broadcast_arrays(r_tensor, phi_tensor)
                rs = math.reshape(rs, (-1,))
                phis = math.reshape(phis, (-1,))
                ret = math.astensor(
                    [math.squeezed(r, phi, shape=shape) for r, phi in zip(rs, phis)],
                )
                ret = math.reshape(ret, ansatz.batch_shape + shape)
                if ansatz._lin_sup:
                    ret = math.sum(ret, axis=ansatz.batch_dims - 1)
            else:
                ret = math.squeezed(r, phi, shape=shape)
            return ret

        self._specialized_fock = specialized_fock

        super().__init__(ansatz=ansatz, wires=wires, name="SqueezedVacuum")

    def fock_array(
        self,
        shape: int | Sequence[int] | None = None,
    ) -> ComplexTensor:
        shape = self._check_fock_shape(shape)
        return self._specialized_fock(shape)
