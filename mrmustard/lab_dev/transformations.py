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
The classes representing transformations in quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

from mrmustard import math
from ..physics.representations import Bargmann
from .circuits import Circuit
from .circuit_components import CircuitComponent
from .utils import make_parameter

__all__ = ["Dgate", "Transformation", "Unitary"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    def __rshift__(self, other: CircuitComponent):
        r"""
        Returns a ``Circuit`` with two components, light-copied from ``self`` and ``other``.
        """
        return Circuit([self, other])


class Unitary(Transformation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this unitary.
        modes: The modes that this unitary acts on.
    """

    def __init__(self, name, modes):
        super().__init__(name, modes_in_ket=modes, modes_out_ket=modes)


class Dgate(Unitary):
    r"""

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats. One can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[Sequence[int]] = None,
    ) -> None:
        m = max(len(math.atleast_1d(x)), len(math.atleast_1d(y)))
        super().__init__("Dgate", modes=modes or list(range(m)))
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = np.array([xs[0] for _ in range(len(self.modes))]) 
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = np.array([ys[0] for _ in range(len(self.modes))]) 

        A = np.array([[0, 1], [1, 0]])
        for _ in range(len(self.modes) - 1):
            A = np.kron(A, A)
        B = math.concat([xs, ys], axis=0)
        C = np.prod([np.exp(-abs(x+1j*y)**2/2) for x, y in zip(xs, ys)])

        return Bargmann(A, B, C)
