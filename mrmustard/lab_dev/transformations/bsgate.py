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
The class representing a beam splitter gate.
"""

from __future__ import annotations

from typing import Sequence

from .base import Unitary
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter

__all__ = ["BSgate"]


class BSgate(Unitary):
    r"""
    The beam splitter gate.

    It applies to a single pair of modes.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import BSgate

        >>> unitary = BSgate(modes=[1, 2], theta=0.1)
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.theta.value, 0.1)
        >>> assert np.allclose(unitary.phi.value, 0.0)

    Args:
        modes: The modes this gate is applied to.
        theta: The transmissivity angle.
        theta_bounds: The bounds for the transmissivity angle.
        theta_trainable: Whether theta is a trainable variable.
        phi: The phase angle.
        phi_bounds: The bounds for the phase angle.
        phi_trainable: Whether phi is a trainable variable.

    Raises:
        ValueError: If ``modes`` is not a pair of modes.

    .. details::

        The beamsplitter gate is a Gaussian gate defined by

        .. math::
            S = \begin{bmatrix}
                    \text{Re}(U) & -\text{Im}(U)\\
                    \text{Im}(U) & \text{Re}(U)
                \end{bmatrix} \text{ and }
            d = O_4\:,

        with

        .. math::
            U &= \begin{bmatrix}
                    \text{cos}(\theta) & -e^{-i\phi}\text{sin}(\theta)\\
                    e^{i\phi}\text{sin}(\theta) & \text{cos}(\theta)
                \end{bmatrix} \\

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = \begin{bmatrix}
                    O_2 & U \\
                    U^{T} & O_2
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = 1
    """

    short_name = "BS"

    def __init__(
        self,
        modes: Sequence[int],
        theta: float = 0.0,
        phi: float = 0.0,
        theta_trainable: bool = False,
        phi_trainable: bool = False,
        theta_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        if len(modes) != 2:
            raise ValueError(f"Expected a pair of modes, found {modes}.")

        super().__init__(modes_out=modes, modes_in=modes, name="BSgate")
        self._add_parameter(make_parameter(theta_trainable, theta, "theta", theta_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.beamsplitter_gate_Abc, theta=self.theta, phi=self.phi
        )
