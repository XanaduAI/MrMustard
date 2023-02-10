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

from typing import Any, List, Optional, Tuple

import numpy as np

from mrmustard.lab.abstract import State, Transformation
from mrmustard.training import Parametrized
from mrmustard.types import Matrix, Vector
from mrmustard.utils.xptensor import XPMatrix, XPVector


class Circuit(Transformation, Parametrized):
    """Represents a quantum circuit: a set of operations to be applied on quantum states.

    Args:
        ops (list or none): A list of operations comprising the circuit.
    """

    def __init__(self, ops: Optional[List] = None):
        self._ops = list(ops) if ops is not None else []
        super().__init__()
        self.reset()
        self.graph = self._graph_from_ops(self._ops)

    def _graph_from_ops(self, ops):
        graph = np.zeros((len(ops), len(ops), self.num_modes), dtype=int)
        for i, op in enumerate(ops):
            for m, mode in enumerate(op.modes):
                for j, op2 in enumerate(ops[i + 1 :]):
                    if mode in op2.modes:
                        graph[i, i + j + 1, mode] = 1
                        graph[i + j + 1, i, mode] = 1
                        break
                for j, op2 in enumerate(reversed(ops[:i])):
                    if mode in op2.modes:
                        graph[i, i - j - 1, mode] = 1
                        graph[i - j - 1, i, mode] = 1
                        break
        return graph

    def _order(self, op1, op2) -> bool:
        "whether the order op1->op2 is right"
        return self._op_index(op1) < self._op_index(op2)

    def _op_connections(self, op):
        "the list of operations connected to op in the form List[(Mode, Op)]"
        connections: List[Tuple[int, Any]] = []
        local_graph = self._graph[:, self._op_index(op), :]  # axes now are [modes, op2_index]
        for op2 in self._ops:
            for i, mode in enumerate(local_graph[:, self._op_index(op2)]):
                if mode:
                    connections.append((i, op2))
        return connections

    def _op_index(self, op):
        "the index of op in the circuit"
        return self._ops.index(op)

    def get_wire_connections(self):
        connections = []
        disconnected_wires = []
        for i, op1 in enumerate(self._ops):
            op1_modes = set(op1.modes)
            print(op1_modes)
            for mode in op1.modes:
                for j, op2 in enumerate(self._ops[i + 1 :]):
                    if mode in op2.modes:
                        connections.append(((i, mode), (j + i + 1, mode)))
                        op1_modes.remove(mode)
                        break
            for mode in op1_modes:
                disconnected_wires.append((i, mode))
        return connections, disconnected_wires

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

    @property
    def is_unitary(self):
        """Returns `true` if all operations in the circuit are unitary."""
        return all(op.is_unitary for op in self._ops)

    def __len__(self):
        return len(self._ops)

    def _repr_markdown_(self) -> str:
        """Markdown string to display the object on ipython notebooks."""
        header = f"#### Circuit  -  {len(self._ops)} ops  -  compiled = `{self._compiled}`\n\n"
        ops_repr = [op._repr_markdown_() for op in self._ops]  # pylint: disable=protected-access
        return header + "\n".join(ops_repr)

    def __repr__(self) -> str:
        """String to display the object on the command line."""
        ops_repr = [repr(op) for op in self._ops]
        return " >> ".join(ops_repr)

    def __str__(self):
        """String representation of the circuit."""
        return f"< Circuit | {len(self._ops)} ops | compiled = {self._compiled} >"


class TokenDispenser:
    r"""A singleton class that generates unique tokens (ints).
    It can be given back tokens to reuse them.

    Example:
        >>> dispenser = TokenDispenser()
        >>> dispenser.get()
        0
        >>> dispenser.get()
        1
        >>> dispenser.give_back(0)
        >>> dispenser.get()
        0
        >>> dispenser.get()
        2
    """
    _instance = None

    def __new__(cls):
        if TokenDispenser._instance is None:
            TokenDispenser._instance = object.__new__(cls)
        return TokenDispenser._instance

    def __init__(self):
        self._tokens = []

    def get(self) -> int:
        """Returns a new token."""
        if self._tokens:
            return self._tokens.pop()
        return len(self._tokens)

    def give_back(self, token: int):
        """Gives back a token to be reused."""
        self._tokens.append(token)

    def reset(self):
        """Resets the dispenser."""
        self._tokens = []

    def __len__(self):
        return len(self._tokens)

    def __repr__(self):
        return f"< TokenDispenser | {len(self._tokens)} tokens >"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        return bool(self._tokens)
