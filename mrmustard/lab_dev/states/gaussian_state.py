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
Classes representing Gaussian states.
"""

from __future__ import annotations

from typing import Sequence

from mrmustard import math
from mrmustard.math.parameters import update_symplectic
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples
from mrmustard.utils.typing import RealMatrix
from mrmustard.lab_dev.graph_component_utils import TraceOut
from .ket import Ket
from .dm import DM
from ..utils import make_parameter, reshape_params

__all__ = ["GKet", "GDM"]


class GKet(Ket):
    r"""
    The `N`-mode pure state described by a Gaussian gate that acts on Vacuum.

    ..details:
        For a given Gaussian unitary U (that is determined by its symplectic
        representation), produces the state
        ..math:
            |\psi\rangle = U |0\rangle

    Args:
        modes: the modes over which the state is defined.

        symplectic: the symplectic representation of the unitary that acts on
        vacuum to produce the desired state. If `None`, a random symplectic matrix
        is chosen.

        symplectic_trainable: determines if the symplectic matrix can be trained.

    """

    short_name = "Gk"

    def __init__(
        self,
        modes: Sequence[int],
        symplectic: RealMatrix = None,
        symplectic_trainable: bool = False,
    ) -> None:
        super().__init__(name="GKet")
        m = len(modes)

        symplectic = symplectic if symplectic is not None else math.random_symplectic(m)

        self.parameters.add_parameter(
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
                fn=triples.gket_state_Abc, symplectic=self.parameters.symplectic
            ),
        ).representation

    def _getitem_builtin(self, modes: set[int] | Sequence[int]):
        r"""
        The slicing method for a GDM state.
        """

        remaining_modes = [mode for mode in self.modes if mode not in modes]

        return self >> TraceOut(remaining_modes)


class GDM(DM):
    r"""
    The `N`-mode mixed state described by a Gaussian gate that acts on a given
    thermal state.

    ..details:
        For a given Gaussian unitary U (that is determined by its symplectic
        representation), and a set of temperatures, produces the state
        ..math:
            \rho = U (\bigotimes_i \rho_t(\beta_i))
        where rho_t are thermal states with temperatures determined by beta.

    Args:
        modes: the modes over which the state is defined.

        beta: the set of temperatures determining the thermal states. If only a
        float is provided for a multi-modes, the same temperature is considered
        across all modes.

        symplectic: the symplectic representation of the unitary that acts on
        vacuum to produce the desired state. If `None`, a random symplectic matrix
        is chosen.

        symplectic_trainable: determines if the symplectic matrix can be trained.
    """

    short_name = "Gd"

    def __init__(
        self,
        modes: Sequence[int],
        beta: float | Sequence[float],
        symplectic: RealMatrix = None,
        symplectic_trainable: bool = False,
        beta_trainable: bool = False,
    ) -> None:
        super().__init__(name="GDM")
        m = len(modes)
        symplectic = symplectic if symplectic is not None else math.random_symplectic(m)
        (betas,) = list(reshape_params(len(modes), betas=beta))
        self.parameters.add_parameter(
            make_parameter(
                symplectic_trainable,
                symplectic,
                "symplectic",
                (None, None),
                update_symplectic,
            )
        )
        self.parameters.add_parameter(
            make_parameter(
                beta_trainable,
                betas,
                "beta",
                (0, None),
            )
        )
        self._representation = self.from_ansatz(
            modes=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.gdm_state_Abc,
                betas=self.parameters.beta,
                symplectic=self.parameters.symplectic,
            ),
        ).representation

    def _getitem_builtin(self, modes: set[int] | Sequence[int]):
        r"""
        The slicing method for a GDM state.
        """
        remaining_modes = [mode for mode in self.modes if mode not in modes]

        return self >> TraceOut(remaining_modes)
