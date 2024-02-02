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
The classes representing states in quantum circuits.
"""

from __future__ import annotations

from typing import Sequence, Optional, Tuple, Union

from mrmustard import math
from ..physics.representations import Bargmann
from ..utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Mode
from .circuits import Circuit
from .circuit_components import CircuitComponent
from .state import Ket, DM
from mrmustard.lab.utils import make_parameter
__all__ = ["Pure", "State", "Vacuum"]


class State(CircuitComponent):
    r"""
    Base class for all states.
    """

    def __rshift__(self, other: CircuitComponent):
        raise NotImplementedError


class Pure(State):
    r"""
    Base class for all pure states.

    Arguments:
        name: The name of this pure state.
        modes: The modes of this pure states.
    """

    def __init__(self, name: str, modes: Sequence[Mode]):
        super().__init__(name, modes_out_ket=modes)


class Vacuum(Pure):
    r"""
    The N-mode vacuum state.

    Args:
        num_modes: the number of modes.
    """

    def __init__(
        self,
        num_modes: int,
    ) -> None:
        super().__init__("Vacuum", modes=list(range(num_modes)))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)
        A = math.zeros(shape=(num_modes, num_modes), dtype=math.complex128)
        B = math.zeros(shape=(num_modes), dtype=math.complex128)
        C = 1.0
        return Bargmann(A, B, C)


class Coherent(Ket):

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
        num_modes = len(self.modes)

        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = math.astensor([xs[0] for _ in range(num_modes)])
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = math.astensor([ys[0] for _ in range(num_modes)])

        A = math.zeros(shape=(num_modes, num_modes), dtype=math.complex128)
        b = math.make_complex(xs, ys)
        c = math.exp(-0.5 * (math.norm(b) ** 2)) + 0.0j

        return Bargmann(A, b, c)
