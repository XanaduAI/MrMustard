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

from typing import Sequence

import numpy as np

from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples
from .base import Ket
from ..utils import make_parameter, reshape_params

__all__ = ["QuadratureEigenstate"]


class QuadratureEigenstate(Ket):
    r"""
    The `N`-mode Quadrature eigenstate.

    .. code-block ::

        >>> from mrmustard.lab_dev import QuadratureEigenstate

        >>> state = QuadratureEigenstate([1, 2], x = 1, phi = 0)
        >>> assert state.modes == [1, 2]

    Args:
        modes: A list of modes.
        x: The displacement of the state.
        phi: The angle of the state with `0` being a position eigenstate and `\pi/2` being the momentum eigenstate.

    .. details::
        Its ``(A,b,c)`` triple is given by

        .. math::
            A = -I_{N}\exp(i2\phi)\text{, }b = I_Nx\exp(i\phi)\sqrt{2/\hbar}\text{, and }c = 1/(\pi\hbar)^{-1/4}\exp(-\abs{x}^2/(2\hbar)).
    """

    def __init__(
        self,
        modes: Sequence[int],
        x: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
        x_trainable: bool = False,
        phi_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(modes=modes, name="QuadratureEigenstate")
        xs, phis = list(reshape_params(len(modes), x=x, phi=phi))
        self._add_parameter(make_parameter(x_trainable, xs, "x", x_bounds))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))
        self._representation = Bargmann.from_function(
            fn=triples.quadrature_eigenstates_Abc, x=self.x, phi=self.phi
        )

        self.manual_shape = (50,)

    @property
    def L2_norm(self):
        r"""
        The L2 norm of this quadrature eigenstate.
        """
        return np.inf
