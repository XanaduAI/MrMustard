# Copyright 2024 Xanadu Quantum Technologies Inc.

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
The class representing a quadratic phase gate.
"""

from __future__ import annotations
from typing import Sequence
from mrmustard.physics.ansatz import PolyExpAnsatz

from .base import Unitary
from ..utils import make_parameter
from ...physics import symplectics

__all__ = ["Pgate"]


class Pgate(Unitary):
    r"""
    Quadratic phase gate.

    Args:
        modes: The modes this gate is applied to.
        shearing: The shearing parameter.
        shearing_trainable: Whether ``shearing`` is trainable.
        shearing_bounds: The bounds for ``shearing``.

    .. details::
        The quadratic phase gate is defined as

        .. math::

            P = \exp(i s q^2 / 2 \hbar)

    Reference: https://strawberryfields.ai/photonics/conventions/gates.html
    """

    short_name = "P"

    def __init__(
        self,
        mode: int,
        shearing: float | Sequence[float] = 0.0,
        shearing_trainable: bool = False,
        shearing_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="Pgate")
        self.parameters.add_parameter(
            make_parameter(shearing_trainable, shearing, "shearing", shearing_bounds)
        )
        self._representation = self.from_ansatz(
            modes_in=(mode,),
            modes_out=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda shearing: Unitary.from_symplectic(
                    (mode,), symplectics.pgate_symplectic(1, shearing)
                ).bargmann_triple(),
                shearing=self.parameters.shearing,
            ),
        ).representation
