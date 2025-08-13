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
The class representing a quadrature eigenstate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import ReprEnum, Wires

from .ket import Ket

__all__ = ["QuadratureEigenstate"]


class QuadratureEigenstate(Ket):
    r"""
    The Quadrature eigenstate in Bargmann representation.

    Args:
        mode: The mode of the quadrature eigenstate.
        x: The displacement of the state.
        phi: The angle of the state with `0` being a position eigenstate and `\pi/2` being the momentum eigenstate.

    .. code-block::

        >>> from mrmustard.lab import QuadratureEigenstate

        >>> state = QuadratureEigenstate(1, x = 1, phi = 0)
        >>> assert state.modes == (1,)

    .. details::
        Its ``(A,b,c)`` triple is given by

        .. math::
            A = -I_{N}\exp(i2\phi)\text{, }b = I_Nx\exp(i\phi)\sqrt{2/\hbar}\text{, and }c = 1/(\pi\hbar)^{-1/4}\exp(-\abs{x}^2/(2\hbar)).
    """

    def __init__(
        self,
        mode: int | tuple[int],
        x: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="QuadratureEigenstate")

        self.parameters.add_parameter(x, "x")
        self.parameters.add_parameter(phi, "phi")

        A, b, c = triples.quadrature_eigenstates_Abc(
            x=self.parameters.x.value,
            phi=self.parameters.phi.value,
        )
        self._ansatz = PolyExpAnsatz(A, b, c)
        self._wires = Wires(modes_out_ket=set(mode))

        for w in self.wires.sorted_wires:
            w.repr = ReprEnum.QUADRATURE
            w.fock_cutoff = 50

    @property
    def L2_norm(self):
        r"""
        The L2 norm of this quadrature eigenstate.
        """
        return np.inf
