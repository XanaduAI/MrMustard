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

from typing import List, Tuple

from mrmustard import settings
from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.lab.abstract.operation import Operation
from mrmustard.training import Parametrized
from mrmustard.types import Matrix, Vector
from mrmustard.utils.circdrawer import circuit_text
from mrmustard.utils.xptensor import XPMatrix, XPVector


class Circuit(Parametrized):
    """Represents a quantum circuit: a set of operations to be applied on quantum states.

    Args:
        in_modes (list[int]): list of input modes
        out_modes (list[int]): list of output modes
        layers (list[list[Operation]]): list of layers, where each layer is a list of operations
        name (str): name of the circuit
        double_wires (bool): whether to double the wires in the circuit
    """

    def __init__(
        self,
        in_modes: List[int] = [],
        out_modes: List[int] = [],
        layers: List[List[Operation]] = [[]],
        name: str = "",
        double_wires: bool = False,
    ):
        OUT = len(out_modes)
        IN = len(in_modes)
        self.dispenser = TagDispenser()

        self.tags: Dict[str, List[int]] = {
            "out_L": [self.dispenser.get_tag() for _ in range(OUT)],
            "in_L": [self.dispenser.get_tag() for _ in range(IN)],
            "out_R": [self.dispenser.get_tag() for _ in range(self.LR * OUT)],
            "in_R": [self.dispenser.get_tag() for _ in range(self.LR * IN)],
        }
        self.wires: Dict[str, List[int]] = {
            "out_L": [i for i in range(OUT)],
            "in_L": [i + OUT for i in range(IN)],
            "out_R": [i + OUT + IN for i in range(self.LR * OUT)],
            "in_R": [i + 2 * OUT + IN for i in range(self.LR * IN)],
        }
        self.layers = layers
        self._compiled: bool = False
        if self.set_double_wires():
            self.double_wires_all_ops()
            self.double_wires = True
        else:
            self.double_wires = double_wires
        self.connect_layers()
        super().__init__()

    def set_double_wires(self) -> bool:
        double_wires = False
        for layer in self.layers:
            for op in layer:
                if op.double_wires:
                    double_wires = True
                    break
            if double_wires:
                break
        return double_wires

    def double_wires_all_ops(self):
        for layer in self.layers:
            for op in layer:
                op.double_wires = True

    def connect(self, mode: int, other: Circuit):
        axes1 = self.mode_to_axes(mode)
        axes2 = other.mode_to_axes(mode)
        for ax1, ax2 in zip(axes1, axes2):
            min_tag = min(other.tags[ax2], self.tags[ax1])
            max_tag = max(other.tags[ax2], self.tags[ax1])
            self.dispenser.give_back_tag(max_tag)
            other.tags[ax2] = min_tag
            self.tags[ax1] = min_tag

    def connect_layers(self):
        "set wire connections for TN contractions or phase space products"
        # NOTE: if double_wires is True for one op, then it must be for all ops. Revisit this at some point.

        for i, layeri in enumerate(self.layers):
            for j, layerj in enumerate(self.layers[i + 1 :]):
                for op1 in layeri:
                    for op2 in layerj:
                        for mode in set(op1.modes_out) & set(op2.modes_in):
                            axes1 = op1.mode_to_axes(mode)
                            axes2 = op2.mode_to_axes(mode)
                            for ax1, ax2 in zip(axes1, axes2):
                                min_tag = min(op2.tags[ax2], op1.tags[ax1])
                                op2.tags[ax2] = min_tag
                                op1.tags[ax1] = min_tag

    # @property
    # def num_modes(self) -> int:
    #     all_modes = {mode for op in self._ops for mode in op.modes}
    #     return len(all_modes)

    # def primal(self, state: State) -> State:
    #     for op in self._ops:
    #         state = op.primal(state)
    #     return state

    # def dual(self, state: State) -> State:
    #     for op in reversed(self._ops):
    #         state = op.dual(state)
    #     return state

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
    ) -> Tuple[Matrix, Matrix, Vector]:  # NOTE: Overriding Transformation.XYd for efficiency.
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

    _repr_markdown_ = None

    def __repr__(self) -> str:
        """String to display the object on the command line."""
        return circuit_text(self._ops, decimals=settings.CIRCUIT_DECIMALS)

    def __str__(self):
        """String representation of the circuit."""
        return f"< Circuit | {len(self._ops)} ops | compiled = {self._compiled} >"
