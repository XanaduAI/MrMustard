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
        x: The displacements along the ``x`` axis, which represents the position axis in phase space.
        y: The displacements along the ``y`` axis, which represents the momentum axis in phase space.
        x_trainable: Whether ``x`` is a trainable variable.
        y_trainable: Whether ``y`` is a trainable variable.
        x_bounds: The bounds for ``x``.
        y_bounds: The bounds for ``y``.

    .. code-block::

        >>> from mrmustard.lab import Dgate

        >>> unitary = Dgate(mode=1, x=0.1, y=0.2)
        >>> assert unitary.modes == (1,)
        >>> assert unitary.parameters.x.value == 0.1
        >>> assert unitary.parameters.y.value == 0.2

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
        mode: int | tuple[int],
        x: float | Sequence[float] = 0.0,
        y: float | Sequence[float] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        y_bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Dgate")
        self.parameters.add_parameter(
            make_parameter(x_trainable, x, "x", x_bounds, dtype=math.float64)
        )
        self.parameters.add_parameter(
            make_parameter(y_trainable, y, "y", y_bounds, dtype=math.float64)
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.displacement_gate_Abc,
            x=self.parameters.x,
            y=self.parameters.y,
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
        """
        if isinstance(shape, int):
            shape = (shape,) * self.ansatz.num_vars
        auto_shape = self.auto_shape()
        shape = shape or auto_shape
        shape = tuple(shape)
        if len(shape) != len(auto_shape):
            raise ValueError(
                f"Expected Fock shape of length {len(auto_shape)}, got length {len(shape)}",
            )
        if self.ansatz.batch_shape:
            x, y = math.broadcast_arrays(self.parameters.x.value, self.parameters.y.value)
            x = math.reshape(x, (-1,))
            y = math.reshape(y, (-1,))
            ret = math.astensor([math.displacement(xi, yi, shape=shape) for xi, yi in zip(x, y)])
            ret = math.reshape(ret, self.ansatz.batch_shape + shape)
            if self.ansatz._lin_sup:
                ret = math.sum(ret, axis=self.ansatz.batch_dims - 1)
        else:
            ret = math.displacement(self.parameters.x.value, self.parameters.y.value, shape=shape)
        return ret
