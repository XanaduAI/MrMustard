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
from ..utils import make_parameter
from .base import Unitary

__all__ = ["Dgate"]


class Dgate(Unitary):
    r"""
    The displacement gate.


    Args:
        mode: The mode this gate is applied to.
        alpha: The displacement in the complex phase space.
        alpha_trainable: Whether ``alpha`` is a trainable variable.
        alpha_bounds: The bounds for the absolute value of ``alpha``.

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
        alpha_trainable: bool = False,
        alpha_bounds: tuple[float | None, float | None] = (0, None),
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Dgate")
        self.parameters.add_parameter(
            make_parameter(alpha_trainable, alpha, "alpha", alpha_bounds, dtype=math.complex128),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.displacement_gate_Abc,
            alpha=self.parameters.alpha,
        )
        self._wires = Wires(set(), set(), set(mode), set(mode))

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
        ret = math.displacement(self.parameters.alpha.value, shape=shape)
        if self.ansatz._lin_sup:
            ret = math.sum(ret, axis=self.ansatz.batch_dims - 1)
        return ret
