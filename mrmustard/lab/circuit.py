# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module implements the :class:`.Circuit` class which acts as a representation for quantum circuits.
"""

from __future__ import annotations

__all__ = ["Circuit"]


from typing import List, Tuple, Optional
from mrmustard.types import Matrix, Vector
from mrmustard.utils.parametrized import Parametrized
from mrmustard.utils.xptensor import XPMatrix, XPVector
from mrmustard.lab.abstract import Transformation
from mrmustard.lab.abstract import State


class Circuit(Transformation, Parametrized):
    """Represents a quantum circuit: a set of operations to be applied on quantum states.

    Args:
        ops (list or none): A list of operations comprising the circuit.
    """

    def __init__(self, ops: Optional[List] = None):
        self._ops = list(ops) if ops is not None else []
        super().__init__()
        self.reset()

    def reset(self):
        """Resets the state of the circuit clearing the list of modes and setting the compiled flag to false."""
        self._compiled: bool = False
        self._modes: List[int] = []

    @property
    def num_modes(self) -> int:
        all_modes = {mode for op in self._ops for mode in op.modes}
        return len(all_modes)

    def primal(self, state: State) -> State:
        for op in self._ops:
            state = op.primal(state)
        return state

    def dual(self, state: State) -> State:
        for op in reversed(self._ops):
            state = op.dual(state)
        return state

    @property
    def XYd(
        self,
    ) -> Tuple[Matrix, Matrix, Vector]:  # NOTE: Overriding Transformation.XYd for efficiency
        X = XPMatrix(like_1=True)
        Y = XPMatrix(like_0=True)
        d = XPVector()
        for op in self._ops:
            opx, opy, opd = op.XYd
            opX = XPMatrix.from_xxpp(opx, modes=(op.modes, op.modes), like_1=True)
            opY = XPMatrix.from_xxpp(opy, modes=(op.modes, op.modes), like_0=True)
            opd = XPVector.from_xxpp(opd, modes=op.modes)
            if opX.shape is not None and opX.shape[-1] == 1 and len(op.modes) > 1:
                opX = opX.clone(len(op.modes), modes=(op.modes, op.modes))
            if opY.shape is not None and opY.shape[-1] == 1 and len(op.modes) > 1:
                opY = opY.clone(len(op.modes), modes=(op.modes, op.modes))
            if opd.shape is not None and opd.shape[-1] == 1 and len(op.modes) > 1:
                opd = opd.clone(len(op.modes), modes=op.modes)
            X = opX @ X
            Y = opX @ Y @ opX.T + opY
            d = opX @ d + opd
        return X.to_xxpp(), Y.to_xxpp(), d.to_xxpp()

    @property
    def is_gaussian(self):
        """Returns `true` if all operations in the circuit are Gaussian."""
        return all(op.is_gaussian for op in self._ops)

    def __len__(self):
        return len(self._ops)

    def _repr_markdown_(self) -> str:
        """Markdown string to display the object on ipython notebooks."""
        return f"Circuit | {len(self._ops)} ops | compiled = `{self._compiled}`"

    def __repr__(self) -> str:
        """String to display the object on the command line."""
        ops_repr = [repr(op) for op in self._ops]
        return "Circuit([" + ",".join(ops_repr) + "])"
