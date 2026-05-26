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

"""The class representing a quadrature eigenstate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mrmustard import math
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import quadrature_eigenstate
from .ket import Ket

__all__ = ["QuadratureEigenstate"]


class QuadratureEigenstate(Ket):
    r"""The Quadrature eigenstate in Bargmann representation.

    >>> from mrmustard.lab import QuadratureEigenstate
    >>> state = QuadratureEigenstate(1, x = 1, phi = 0)
    >>> assert state.modes == (1,)

    Args:
        mode: The mode of the quadrature eigenstate.
        x: The displacement of the state.
        phi: The angle of the state with `0` being a position eigenstate and `\pi/2` being the momentum eigenstate.
        name: A name for the state. If not provided, the class name will be used.

    .. details::
        Its ``(A,b,c)`` triple is given by

        .. math::
            A = -I_{N}\exp(i2\phi)\text{, }b = I_Nx\exp(i\phi)\sqrt{2/\hbar}\text{, and }c = 1/(\pi\hbar)^{-1/4}\exp(-\abs{x}^2/(2\hbar)).
    """

    short_name = "Qe"

    def __init__(
        self,
        mode: int | tuple[int],
        x: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (quadrature_eigenstate, ("x", "phi", "lin_sup"))}
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["x"] = Parameter.from_cc_init(x, "float64", f"{self.name}/x")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")

        for w in self.wires.standard_order:
            w.repr = ReprEnum.QUADRATURE
            w.fock_shape = 50

    @property
    def L2_norm(self):
        r"""The L2 norm of this quadrature eigenstate."""
        return math.full(self.ansatz.batch_shape, np.inf)
