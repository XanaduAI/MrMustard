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

from .base import Unitary
from ...physics.ansatz import PolyExpAnsatz
from ...physics import triples_batched
from ..utils import make_parameter

__all__ = ["Rgate"]


class Rgate(Unitary):
    r"""
    The rotation gate.

    .. code-block ::

        >>> from mrmustard.lab_dev import Rgate

        >>> unitary = Rgate(mode=1, theta=0.1)
        >>> assert unitary.modes == (1,)

    Args:
        mode: The mode this gate is applied to.
        theta: The rotation angle.
        theta_trainable: Whether ``theta`` is trainable.
        theta_bounds: The bounds for ``theta``.
    """

    short_name = "R"

    def __init__(
        self,
        mode: int,
        theta: float = 0.0,
        theta_trainable: bool = False,
        theta_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        super().__init__(name="Rgate")
        self.parameters.add_parameter(make_parameter(theta_trainable, theta, "theta", theta_bounds))
        self._representation = self.from_ansatz(
            modes_in=(mode,),
            modes_out=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=triples_batched.rotation_gate_Abc, theta=self.parameters.theta
            ),
        ).representation
