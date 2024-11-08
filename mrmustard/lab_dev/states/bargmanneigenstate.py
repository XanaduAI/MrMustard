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
The class representing a Bargmann eigenstate.
"""

from __future__ import annotations

from typing import Sequence

from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples
from .ket import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["BargmannEigenstate"]


class BargmannEigenstate(Ket):
    r"""
    Multimode Bargmann eigenstate. These are basically re-scaled coherent states i.e.,
    .. math::
        A = 0 , b = alpha, c = 1
    """

    short_name = "Be"

    def __init__(
        self,
        modes: Sequence[int],
        alpha: float | Sequence[float] = 0.0,
        alpha_trainable: bool = False,
        alpha_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="BargmannEigenstate")

        self._add_parameter(make_parameter(alpha_trainable, alpha, "alpha", alpha_bounds))
        self._representation = self.from_ansatz(
            modes=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.bargmann_eigenstate_Abc, x=self.alpha.value
            ),
        ).representation
