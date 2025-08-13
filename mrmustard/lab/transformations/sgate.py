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

from mrmustard import math
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexTensor

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from .base import Unitary

__all__ = ["Sgate"]


class Sgate(Unitary):
    r"""
    The squeezing gate.


    Args:
        mode: The mode this gate is applied to.
        r: The squeezing magnitude.
        phi: The squeezing angle.

    .. code-block::

        >>> from mrmustard.lab import Sgate

        >>> unitary = Sgate(mode=1, r=0.1, phi=0.2)
        >>> assert unitary.modes == (1,)
        >>> assert unitary.parameters.r.value == 0.1
        >>> assert unitary.parameters.phi.value == 0.2

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
        r: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
    ):
        # Store parameters privately for fock_array method
        self._r = r
        self._phi = phi
        
        A, b, c = triples.squeezing_gate_Abc(r=r, phi=phi)
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_ket={mode}, modes_in_ket={mode})
        
        super().__init__(ansatz=ansatz, wires=wires, name="Sgate")

    def fock_array(
        self,
        shape: int | Sequence[int] | None = None,
    ) -> ComplexTensor:
        shape = self._check_fock_shape(shape)
        if self.ansatz.batch_shape:
            rs, phi = math.broadcast_arrays(
                self._r,
                self._phi,
            )
            rs = math.reshape(rs, (-1,))
            phi = math.reshape(phi, (-1,))
            ret = math.astensor(
                [math.squeezer(r, p, shape=shape) for r, p in zip(rs, phi)],
            )
            ret = math.reshape(ret, self.ansatz.batch_shape + shape)
            if self.ansatz._lin_sup:
                ret = math.sum(ret, axis=self.ansatz.batch_dims - 1)
        else:
            ret = math.squeezer(
                self._r,
                self._phi,
                shape=shape,
            )
        return ret
