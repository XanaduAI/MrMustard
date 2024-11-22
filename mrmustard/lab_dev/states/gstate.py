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
Classes that are initialized by a symplectic matrix and possibly thermal states if we have a DM.
"""

from __future__ import annotations

from typing import Sequence

from mrmustard import math
from mrmustard.math.parameters import update_symplectic
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples
from mrmustard.utils.typing import RealMatrix
from .ket import Ket
from .dm import DM
from ..utils import make_parameter

__all__ = ["Gket", "Gdm"]


class Gket(Ket):
    r"""
    The `N`-mode pure state described by a Gaussian gate that acts on Vacuum.
    """

    short_name = "Gk"

    def __init__(
        self,
        modes: Sequence[int],
        symplectic: RealMatrix = None,
        symplectic_trainable: bool = False,
    ) -> None:
        super().__init__(name="Gket")
        m = len(modes)
        if symplectic is None:
            symplectic = math.random_symplectic(m)

        self._add_parameter(
            make_parameter(
                symplectic_trainable,
                symplectic,
                "symplectic",
                (None, None),
                update_symplectic,
            )
        )

        self._representation = self.from_ansatz(
            modes=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.gket_state_Abc, symplectic=self.symplectic
            ),
        ).representation


class Gdm(DM):
    r"""
    The `N`-mode mixed state described by a Gaussian gate that acts on a given thermal state.
    """

    short_name = "Gd"

    def __init__(
        self,
        modes: Sequence[int],
        betas: float | Sequence[float],
        symplectic: RealMatrix = None,
        symplectic_trainable: bool = False,
        betas_trainable: bool = False,
    ) -> None:
        super().__init__(name="Gdm")
        m = len(modes)

        if symplectic is None:
            symplectic = math.random_symplectic(m)

        self._add_parameter(
            make_parameter(
                symplectic_trainable,
                symplectic,
                "symplectic",
                (None, None),
                update_symplectic,
            )
        )

        self._add_parameter(
            make_parameter(
                betas_trainable,
                math.astensor(betas),
                "betas",
                (0, None),
            )
        )

        self._representation = self.from_ansatz(
            modes=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.gdm_state_Abc, betas=self.betas, symplectic=symplectic
            ),
        ).representation
