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

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Unitary
from .builtins import squeezing_gate, squeezing_gate_fock

__all__ = ["Sgate"]


class Sgate(Unitary):
    r"""
    The squeezing gate.

    >>> from mrmustard.lab import Sgate
    >>> unitary = Sgate(mode=1, r=0.1, phi=0.2)
    >>> assert unitary.modes == (1,)
    >>> assert unitary.parameters.r.value == 0.1
    >>> assert unitary.parameters.phi.value == 0.2

    Args:
        mode: The mode this gate is applied to.
        r: The squeezing magnitude.
        phi: The squeezing angle.

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
        mode: int | tuple[int],
        r: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (squeezing_gate, ("r", "phi", "lin_sup")),
                    ReprEnum.FOCK: (squeezing_gate_fock, ("r", "phi", "shape", "lin_sup")),
                }
            ),
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["r"] = Parameter.from_cc_init(r, "float64", f"{self.name}/r")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
