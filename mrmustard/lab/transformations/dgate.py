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
The class representing a displacement gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.utils.typing import ComplexTensor

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from ...physics.wires import Wires
from .base import Unitary

__all__ = ["Dgate"]


class Dgate(Unitary):
    r"""
    The displacement gate.


    Args:
        mode: The mode this gate is applied to.
        alpha: The displacement in the complex phase space.

    .. code-block::

        >>> from mrmustard.lab import Dgate

        >>> unitary = Dgate(mode=1, alpha=0.1 + 0.2j)
        >>> assert unitary.modes == (1,)
        >>> assert unitary.parameters.alpha.value == 0.1 + 0.2j

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        displacement gate is defined by

        .. math::
            S = I_N \text{ and } r = \sqrt{2\hbar}\big[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})\big].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    O_N & I_N\\
                    I_N & O_N
                \end{bmatrix} \\ \\
            b &= \begin{bmatrix}
                    \bar{\alpha} & -\bar{\alpha}^*
                \end{bmatrix} \\ \\
            c &= \text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    short_name = "D"

    def __init__(
        self,
        mode: int,
        alpha: complex | Sequence[complex] = 0.0j,
    ) -> None:
        A, b, c = triples.displacement_gate_Abc(alpha=alpha)
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket={mode}, modes_out_ket={mode})
        
        # Create specialized closure that captures alpha
        def specialized_fock(shape, **kwargs):
            """Optimized Fock computation using displacement formula."""
            if ansatz.batch_shape:
                alpha_local = alpha
                alpha_local = math.reshape(alpha_local, (-1,))
                ret = math.astensor([math.displacement(alpha_i, shape=shape) for alpha_i in alpha_local])
                ret = math.reshape(ret, ansatz.batch_shape + shape)
                if ansatz._lin_sup:
                    ret = math.sum(ret, axis=ansatz.batch_dims - 1)
            else:
                ret = math.displacement(alpha, shape=shape)
            return ret
        
        self._specialized_fock = specialized_fock
        
        super().__init__(ansatz=ansatz, wires=wires, name="Dgate")

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        r"""
        Returns the unitary representation of the Displacement gate using the Laguerre polynomials.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it defaults to
                ``settings.DEFAULT_FOCK_SIZE``.

        Returns:
            array: The Fock representation of this component.

        Raises:
            ValueError: If the shape is not valid for the component.
        """
        shape = self._check_fock_shape(shape)
        
        return self._specialized_fock(shape)
