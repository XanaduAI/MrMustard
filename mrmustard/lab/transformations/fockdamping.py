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
from ..utils import make_parameter
from .base import Operation

__all__ = ["FockDamping"]


class FockDamping(Operation):
    r"""
    The Fock damping operator.


    Args:
        mode: The mode this gate is applied to.
        damping: The damping parameter.
        damping_trainable: Whether ``damping`` is trainable.
        damping_bounds: The bounds for ``damping``.

    .. code-block::

        >>> from mrmustard.lab import FockDamping, Coherent

        >>> operator = FockDamping(mode=0, damping=0.1)
        >>> input_state = Coherent(mode=0, x=1, y=0.5)
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
        mode: int | tuple[int],
        damping: float | Sequence[float] = 0.0,
        damping_trainable: bool = False,
        damping_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="FockDamping")
        self.parameters.add_parameter(
            make_parameter(
                damping_trainable,
                damping,
                "damping",
                damping_bounds,
                None,
                dtype=math.float64,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.fock_damping_Abc,
            beta=self.parameters.damping,
        )
        self._wires = Wires(modes_in_ket=set(mode), modes_out_ket=set(mode))
