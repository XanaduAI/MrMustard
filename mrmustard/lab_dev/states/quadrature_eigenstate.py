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

import numpy as np

from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics import triples_batched
from mrmustard.physics.wires import ReprEnum
from .ket import Ket
from ..utils import make_parameter

__all__ = ["QuadratureEigenstate"]


class QuadratureEigenstate(Ket):
    r"""
    The Quadrature eigenstate in Bargmann representation.

    .. code-block ::

        >>> from mrmustard.lab_dev import QuadratureEigenstate

        >>> state = QuadratureEigenstate(1, x = 1, phi = 0)
        >>> assert state.modes == (1,)

    Args:
        mode: The mode of the quadrature eigenstate.
        x: The displacement of the state.
        phi: The angle of the state with `0` being a position eigenstate and `\pi/2` being the momentum eigenstate.
        x_trainable: Whether `x` is trainable.
        phi_trainable: Whether `phi` is trainable.
        x_bounds: The bounds of `x`.
        phi_bounds: The bounds of `phi`.

    .. details::
        Its ``(A,b,c)`` triple is given by

        .. math::
            A = -I_{N}\exp(i2\phi)\text{, }b = I_Nx\exp(i\phi)\sqrt{2/\hbar}\text{, and }c = 1/(\pi\hbar)^{-1/4}\exp(-\abs{x}^2/(2\hbar)).
    """

    def __init__(
        self,
        mode: int,
        x: float = 0.0,
        phi: float = 0.0,
        x_trainable: bool = False,
        phi_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="QuadratureEigenstate")

        self.parameters.add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self.parameters.add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))
        self.manual_shape = (50,)

        self._representation = self.from_ansatz(
            modes=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=triples_batched.quadrature_eigenstates_Abc,
                x=self.parameters.x,
                phi=self.parameters.phi,
            ),
        ).representation

        for w in self.representation.wires.output.wires:
            w.repr = ReprEnum.QUADRATURE
            w.repr_params_func = lambda w=w: [
                self.parameters.x.value,
                self.parameters.phi.value,
            ]

    @property
    def L2_norm(self):
        r"""
        The L2 norm of this quadrature eigenstate.
        """
        return np.inf
