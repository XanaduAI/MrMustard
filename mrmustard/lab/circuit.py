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

import opt_einsum

__all__ = ["Circuit"]

from typing import Any, List, Optional, Tuple

import numpy as np

from mrmustard import settings
from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.lab.abstract.operation import Operation
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
        "whether op1 comes before op2 in the circuit"
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

    def _init_TN_connections(self):
        "Initialize the TN connections. Everything is disconnected."
        dispenser = TagDispenser()
        connections = []
        for op in self._ops:
            op = Operation(op)
            connections.append((op, [dispenser.tag() for _ in range(op.num_axes)]))
        dispenser.reset()
        return connections

    def TN_connectivity(self):
        "get wire connections at the TN level (axis1 <-> axis2 for contraction)"
        connections = self._init_TN_connections()
        for i, (op1, tags1) in enumerate(connections):
            for mode in op1.modes_out:
                ax1 = op1.axes_for_mode(mode)
                for j, (op2, tags2) in enumerate(connections[i + 1 :]):
                    if mode in op2.modes_in:
                        ax2 = op2.axes_for_mode(mode)
                        if ax1.ol_ is not None and ax2.il_ is not None:
                            tags2[ax2.il_] = tags1[ax1.ol_]
                        if ax1.or_ is not None and ax2.ir_ is not None:
                            tags2[ax2.ir_] = tags1[ax1.or_]
                        break
        connections = [(self._ops[i], tags) for i, (_, tags) in enumerate(connections)]
        return connections

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

    def shape_specs(self) -> tuple[dict[int, int], list[int]]:
        # Keep track of the shapes for each tag
        tag_shapes = {}
        fock_tags = []

        # Loop through the list of operations
        for op, tag_list in self.TN_connectivity():
            # Check if this operation is a projection onto Fock
            if isinstance(op, Measurement) and hasattr(op.outcome, "_n"):
                # If it is, set the shape for the tags to Fock.
                for i, tag in enumerate(tag_list):
                    tag_shapes[tag] = op.outcome._n[i] + 1
                    fock_tags.append(tag)
            else:
                # If not, get the default shape for this operation
                shape = [50 for _ in range(Operation(op).num_axes)]  # NOTE: just a placeholder

                # Loop through the tags for this operation
                for i, tag in enumerate(tag_list):
                    # If the tag has not been seen yet, set its shape
                    if tag not in tag_shapes:
                        tag_shapes[tag] = shape[i]
                    else:
                        # If the tag has been seen, set its shape to the minimum of the current shape and the previous shape
                        tag_shapes[tag] = min(tag_shapes[tag], shape[i])

        return tag_shapes, fock_tags

    def TN_tensor_list(self) -> list:
        tag_shapes, fock_tags = self.shape_specs()
        # Loop through the list of operations
        tensors_and_tags = []
        for i, (op, tag_list) in enumerate(self.TN_connectivity()):
            # skip Fock measurements
            if isinstance(op, Measurement) and hasattr(op.outcome, "_n"):
                continue
            else:
                # Convert the operation to a tensor with the correct shape
                shape = [tag_shapes[tag] for tag in tag_list]
                if isinstance(op, Measurement):
                    op = op.outcome.ket(shape)
                elif isinstance(op, State):
                    if op.is_pure:
                        op = op.ket(shape)
                    else:
                        op = op.dm(shape)
                elif isinstance(op, Transformation):
                    if op.is_unitary:
                        op = op.U(shape)
                    else:
                        op = op.choi(shape)
                else:
                    raise ValueError("Unknown operation type")

                fock_tag_positions = [tag_list.index(tag) for tag in fock_tags if tag in tag_list]
                slice_spec = [slice(None)] * len(tag_list)
                for tag_pos in fock_tag_positions:
                    slice_spec[tag_pos] = -1
                op = op[tuple(slice_spec)]
                tag_list = [tag for tag in tag_list if tag not in fock_tags]

                # Add the tensor and its tags to the list
                tensors_and_tags.append((op, tag_list))

        return tensors_and_tags

    def contract(self):
        opt_einsum_args = [item for pair in self.TN_tensor_list() for item in pair]
        return opt_einsum.contract(*opt_einsum_args, optimize=settings.OPT_EINSUM_OPTIMIZE)

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


class TagDispenser:
    r"""A singleton class that generates unique tags (ints).
    It can be given back tags to reuse them.

    Example:
        >>> dispenser = TagDispenser()
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
        if TagDispenser._instance is None:
            TagDispenser._instance = object.__new__(cls)
        return TagDispenser._instance

    def __init__(self):
        self._tags = []
        self._counter = 0

    def tag(self) -> int:
        """Returns a new unique tag."""
        if len(self._tags) > 0:
            return self._tags.pop(0)
        else:
            self._counter += 1
            return self._counter - 1

    def give_back(self, tag: int):
        """Gives back a tag to be reused."""
        self._tags.append(tag)

    def reset(self):
        """Resets the dispenser."""
        self._tags = []
        self._counter = 0

    def __repr__(self):
        _next = self._tags[0] if len(self._tags) > 0 else self._counter
        return f"TagDispenser(returned={self._tags}, next={_next})"
