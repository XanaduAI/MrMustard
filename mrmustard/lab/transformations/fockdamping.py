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
The class representing a rotation gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.wires import Wires

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz

from .base import Operation

__all__ = ["FockDamping"]


class FockDamping(Operation):
    r"""
    The Fock damping operator.


    Args:
        mode: The mode this gate is applied to.
        damping: The damping parameter.

    .. code-block::

        >>> from mrmustard.lab import FockDamping, Coherent

        >>> operator = FockDamping(mode=0, damping=0.1)
        >>> input_state = Coherent(mode=0, alpha=1 + 0.5j)
        >>> output_state = input_state >> operator
        >>> assert operator.modes == (0,)
        >>> assert operator.parameters.damping.value == 0.1
        >>> assert output_state.L2_norm < 1

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= e^{-\beta}\begin{bmatrix}
                    O_N & I_N & \\
                    I_N & O_N &

                \end{bmatrix} \\ \\
            b &= O_{2N} \\ \\
            c &= 1\:.
    """

    def __init__(
        self,
        mode: int,
        damping: float | Sequence[float] = 0.0,
    ):
        A, b, c = triples.fock_damping_Abc(beta=damping)
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket={mode}, modes_out_ket={mode})

        super().__init__(ansatz=ansatz, wires=wires, name="FockDamping")
