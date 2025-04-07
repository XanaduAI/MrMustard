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
The class representing a squeezing gate.
"""

from __future__ import annotations


from .base import Unitary
from ...physics.ansatz import PolyExpAnsatz
from ...physics import triples
from ..utils import make_parameter

__all__ = ["Sgate"]


class Sgate(Unitary):
    r"""
    The squeezing gate.

    .. code-block ::

        >>> from mrmustard.lab_dev import Sgate

        >>> unitary = Sgate(mode=1, r=0.1, phi=0.2)
        >>> assert unitary.modes == (1,)
        >>> assert unitary.parameters.r.value == 0.1
        >>> assert unitary.parameters.phi.value == 0.2

    Args:
        mode: The mode this gate is applied to.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        r_trainable: Whether ``r`` is trainable.
        phi_trainable: Whether ``phi`` is trainable.
        r_bounds: The bounds for ``r``.
        phi_bounds: The bounds for ``phi``.

    .. details::

        For any :math:`\bar{r}` and :math:`\bar{\phi}` of length :math:`N`, the :math:`N`-mode
        squeezing gate is defined by

        .. math::
            S = \begin{bmatrix}
                    \text{diag}_N(\text{cosh}(\bar{r})) & \text{diag}_N(e^{-i\bar{\phi}}\text{sinh}(\bar{r}))\\
                    -\text{diag}_N(e^{i\bar{\phi}}\text{sinh}(\bar{r})) & \text{diag}_N(\text{cosh}(\bar{r}))
                \end{bmatrix} \text{ and }
            d = O_{2N},

        where :math:`\text{diag}_N(\bar{a})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{a}`.
        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    -\text{diag}_N(e^{i\bar{\phi}}\text{tanh}(\bar{r})) & \text{diag}_N(\text{sech}(\bar{r}))\\
                    \text{diag}_N(\text{sech}(\bar{r})) & \text{diag}_N(e^{-i\bar{\phi}}\text{tanh}(\bar{r}))
                \end{bmatrix} \\ \\
            b &= O_{2N} \\ \\
            c &= \prod_{i=1}^N\sqrt{\text{sech}{\:r_i}}\:.
    """

    short_name = "S"

    def __init__(
        self,
        mode: int,
        r: float = 0.0,
        phi: float = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: tuple[float | None, float | None] = (0.0, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="Sgate")
        self.parameters.add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self.parameters.add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))
        self._representation = self.from_ansatz(
            modes_in=(mode,),
            modes_out=(mode,),
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.squeezing_gate_Abc,
                r=self.parameters.r,
                phi=self.parameters.phi,
            ),
        ).representation
