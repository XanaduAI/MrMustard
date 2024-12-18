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

from typing import Sequence
from mrmustard.utils.typing import ComplexTensor

from mrmustard import math

from .base import Unitary
from ...physics.representations import Representation
from ...physics.ansatz import PolyExpAnsatz, ArrayAnsatz
from ...physics import triples, fock_utils
from ..utils import make_parameter, reshape_params

__all__ = ["Dgate"]


class Dgate(Unitary):
    r"""
    The displacement gate.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Dgate

        >>> unitary = Dgate(modes=[1, 2], x=0.1, y=[0.2, 0.3])
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.x.value, [0.1, 0.1])
        >>> assert np.allclose(unitary.y.value, [0.2, 0.3])

    Args:
        modes: The modes this gate is applied to.
        x: The displacements along the `x` axis, which represents position axis in phase space.
        y: The displacements along the `y` axis.
        x_bounds: The bounds for the displacement along the `x` axis.
        y_bounds: The bounds for the displacement along the `y` axis, which represents momentum axis in phase space.
        x_trainable: Whether `x` is a trainable variable.
        y_trainable: Whether `y` is a trainable variable.

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
        modes: Sequence[int] = None,
        x: float | Sequence[float] = 0.0,
        y: float | Sequence[float] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: tuple[float | None, float | None] = (None, None),
        y_bounds: tuple[float | None, float | None] = (None, None),
    ) -> None:
        super().__init__(name="Dgate")
        xs, ys = list(reshape_params(len(modes), x=x, y=y))
        self.parameters.add_parameter(make_parameter(x_trainable, xs, "x", x_bounds))
        self.parameters.add_parameter(make_parameter(y_trainable, ys, "y", y_bounds))
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.displacement_gate_Abc, x=self.parameters.x, y=self.parameters.y
            ),
        ).representation

    def fock_array(self, shape: int | Sequence[int] = None, batched=False) -> ComplexTensor:
        r"""
        Returns the unitary representation of the Displacement gate using the Laguerre polynomials.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is estimated.
            batched: Whether the returned representation is batched or not. If ``False`` (default)
                it will squeeze the batch dimension if it is 1.
        Returns:
            array: The Fock representation of this component.
        """
        if isinstance(shape, int):
            shape = (shape,) * self.ansatz.num_vars
        auto_shape = self.auto_shape()
        shape = shape or auto_shape
        if len(shape) != len(auto_shape):
            raise ValueError(
                f"Expected Fock shape of length {len(auto_shape)}, got length {len(shape)}"
            )
        N = self.n_modes
        x = self.parameters.x.value * math.ones(N, dtype=self.parameters.x.value.dtype)
        y = self.parameters.y.value * math.ones(N, dtype=self.parameters.y.value.dtype)

        if N > 1:
            # calculate displacement unitary for each mode and concatenate with outer product
            Ud = None
            for idx, out_in in enumerate(zip(shape[:N], shape[N:])):
                if Ud is None:
                    Ud = fock_utils.displacement(x[idx], y[idx], shape=out_in)
                else:
                    U_next = fock_utils.displacement(x[idx], y[idx], shape=out_in)
                    Ud = math.outer(Ud, U_next)

            array = math.transpose(
                Ud,
                list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2)),
            )
        else:
            array = fock_utils.displacement(x[0], y[0], shape=shape)
        arrays = math.expand_dims(array, 0) if batched else array
        return arrays

    def to_fock(self, shape: int | Sequence[int] | None = None) -> Dgate:
        fock = ArrayAnsatz(self.fock_array(shape, batched=True), batched=True)
        fock._original_abc_data = self.ansatz.triple
        ret = self._getitem_builtin(self.modes)
        ret._representation = Representation(fock, self.wires)
        return ret
