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
The class representing a two-mode squeezing gate.
"""

from __future__ import annotations

from typing import Sequence

from .base import Unitary
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter

__all__ = ["S2gate"]


class S2gate(Unitary):
    r"""
    The two-mode squeezing gate.

    It applies to a single pair of modes.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import S2gate

        >>> unitary = S2gate(modes=[1, 2], r=1)
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.r.value, 1)
        >>> assert np.allclose(unitary.phi.value, 0.0)

    Args:
        modes: The modes this gate is applied to.
        r: The squeezing amplitude.
        r_bounds: The bounds for the squeezing amplitude.
        r_trainable: Whether r is a trainable variable.
        phi: The phase angle.
        phi_bounds: The bounds for the phase angle.
        phi_trainable: Whether phi is a trainable variable.

    Raises:
        ValueError: If ``modes`` is not a pair of modes.

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = \begin{bmatrix}
                    O & e^{i\phi}\tanh(r) & \sech(r) & 0 \\
                    e^{i\phi}\tanh(r) & 0 & 0 & \sech(r) \\
                    \sech(r) & & 0 & 0 e^{i\phi}\tanh(r) \\
                    O & \sech(r) & e^{i\phi}\tanh(r) & 0
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = \sech(r)
    """

    def __init__(
        self,
        modes: Sequence[int],
        r: float = 0.0,
        phi: float = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: tuple[float | None, float | None] = (0, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        if len(modes) != 2:
            raise ValueError(f"Expected a pair of modes, found {modes}.")

        super().__init__(modes_out=modes, modes_in=modes, name="S2gate")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.twomode_squeezing_gate_Abc, r=self.r, phi=self.phi
        )
