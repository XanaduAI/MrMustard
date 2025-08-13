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

from .base import Unitary

__all__ = ["Rgate"]


class Rgate(Unitary):
    r"""
    The rotation gate.


    Args:
        mode: The mode this gate is applied to.
        theta: The rotation angle.

    .. code-block::

        >>> from mrmustard.lab import Rgate

        >>> unitary = Rgate(mode=1, theta=0.1)
        >>> assert unitary.modes == (1,)
    """

    short_name = "R"

    def __init__(
        self,
        mode: int | tuple[int],
        theta: float | Sequence[float] = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Rgate")
        self.parameters.add_parameter(theta, "theta")
        A, b, c = triples.rotation_gate_Abc(
            theta=self.parameters.theta.value,
        )
        self._ansatz = PolyExpAnsatz(A, b, c)
        self._wires = Wires(modes_in_ket=set(mode), modes_out_ket=set(mode))
