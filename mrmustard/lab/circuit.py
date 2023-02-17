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

from typing import Dict, List, Union

from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.utils.tagdispenser import TagDispenser

# class Circuit(Parametrized):
#     r"""Represents a quantum circuit: a wrapper around States, Transformations and Measurements
#     which allows them to be placed in a circuit and connected together.

#     A Circuit can include several operations and return a new Circuit when concatenated with another
#     Circuit. The Circuit can be compiled to a TensorNetwork, which can be used to evaluate the
#     circuit's free tensor indices.

#     A Circuit can be run, sampled, compiled and composed.

#     Args:
#         in_modes (list[int]): list of input modes
#         out_modes (list[int]): list of output modes
#         ops (Union[State, Transformation, Measurement], list[Circuit]): either a single operation
#             or any number of Circuits
#         name (str): name of the circuit
#         dual_wires (bool): whether to include dual wires across the circuit
#     """

#     def __init__(
#         self,
#         in_modes: List[int] = [],
#         out_modes: List[int] = [],
#         ops: Union[State, Transformation, Measurement, list[Circuit]] = [],
#         name: str = "?",
#         dual_wires: bool = False,
#     ):
#         self.dispenser = TagDispenser()
#         self._tags: Dict[str, List[int]] = None
#         self._axes: Dict[str, List[int]] = None

#         self.ops = ops
#         self._compiled: bool = False
#         if self.any_dual_wires():
#             self.set_dual_wires_everywhere()
#             self.dual_wires = True
#         else:
#             self.dual_wires = dual_wires
#         self.connect_layers()
#         super().__init__()

#     @property
#     def tags(self) -> dict[str, list[int]]:
#         if self._tags is None:
#             OUT = len(self.out_modes)
#             IN = len(self.in_modes)
#             self._tags = {
#                 "out_L": [self.dispenser.get_tag() for _ in range(OUT)],
#                 "in_L": [self.dispenser.get_tag() for _ in range(IN)],
#                 "out_R": [self.dispenser.get_tag() for _ in range(self.dual_wires * OUT)],
#                 "in_R": [self.dispenser.get_tag() for _ in range(self.dual_wires * IN)],
#             }
#         return self._tags

#     @property
#     def axes(self) -> dict[str, list[int]]:
#         OUT = len(self.out_modes)
#         IN = len(self.in_modes)
#         if self._axes is None:
#             self._axes = {
#                 "out_L": [i for i in range(OUT)],
#                 "in_L": [i + OUT for i in range(IN)],
#                 "out_R": [i + OUT + IN for i in range(self.dual_wires * OUT)],
#                 "in_R": [i + 2 * OUT + IN for i in range(self.dual_wires * IN)],
#             }
#         return self._axes

#     def any_dual_wires(self) -> bool:
#         "check that at least one op has dual wires"
#         for op in self.ops:
#             if op.dual_wires:
#                 return True
#         return False

#     def set_dual_wires_everywhere(self):
#         for op in self.ops:
#             op.dual_wires = True

#     def connect_ops(self):
#         "set wire connections for TN contractions or phase space products"
#         # If dual_wires is True for one op, then it must be for all ops.
#         # NOTE: revisit this at some point.
#         for i, opi in enumerate(self.ops):
#             for j, opj in enumerate(self.ops[i + 1 :]):
#                 for mode in set(opi.modes_out) & set(opj.modes_in):
#                     axes1 = opi.mode_out_to_axes(mode)
#                     axes2 = opj.mode_in_to_axes(mode)
#                     for ax1, ax2 in zip(axes1, axes2):
#                         min_tag = min(opi.tags[ax1], opj.tags[ax2])
#                         max_tag = max(opi.tags[ax1], opj.tags[ax2])
#                         opi.tags[ax1] = min_tag
#                         opj.tags[ax2] = min_tag
#                         if max_tag != min_tag:
#                             self.dispenser.give_back_tag(max_tag)

#     # def shape_specs(self) -> tuple[dict[int, int], list[int]]:
#     #     # Keep track of the shapes for each tag
#     #     tag_shapes = {}
#     #     fock_tags = []

#     #     # Loop through the list of operations
#     #     for op, tag_list in self.TN_connectivity():
#     #         # Check if this operation is a projection onto Fock
#     #         if isinstance(op, Measurement) and hasattr(op.outcome, "_n"):
#     #             # If it is, set the shape for the tags to Fock.
#     #             for i, tag in enumerate(tag_list):
#     #                 tag_shapes[tag] = op.outcome._n[i] + 1
#     #                 fock_tags.append(tag)
#     #         else:
#     #             # If not, get the default shape for this operation
#     #             shape = [50 for _ in range(Operation(op).num_axes)]  # NOTE: just a placeholder

#     #             # Loop through the tags for this operation
#     #             for i, tag in enumerate(tag_list):
#     #                 # If the tag has not been seen yet, set its shape
#     #                 if tag not in tag_shapes:
#     #                     tag_shapes[tag] = shape[i]
#     #                 else:
#     #                     # If the tag has been seen, set its shape to the minimum of the current shape and the previous shape
#     #                     tag_shapes[tag] = min(tag_shapes[tag], shape[i])

#     #     return tag_shapes, fock_tags

#     # def TN_tensor_list(self) -> list:
#     #     tag_shapes, fock_tags = self.shape_specs()
#     #     # Loop through the list of operations
#     #     tensors_and_tags = []
#     #     for i, (op, tag_list) in enumerate(self.TN_connectivity()):
#     #         # skip Fock measurements
#     #         if isinstance(op, Measurement) and hasattr(op.outcome, "_n"):
#     #             continue
#     #         else:
#     #             # Convert the operation to a tensor with the correct shape
#     #             shape = [tag_shapes[tag] for tag in tag_list]
#     #             if isinstance(op, Measurement):
#     #                 op = op.outcome.ket(shape)
#     #             elif isinstance(op, State):
#     #                 if op.is_pure:
#     #                     op = op.ket(shape)
#     #                 else:
#     #                     op = op.dm(shape)
#     #             elif isinstance(op, Transformation):
#     #                 if op.is_unitary:
#     #                     op = op.U(shape)
#     #                 else:
#     #                     op = op.choi(shape)
#     #             else:
#     #                 raise ValueError("Unknown operation type")

#     #             fock_tag_positions = [tag_list.index(tag) for tag in fock_tags if tag in tag_list]
#     #             slice_spec = [slice(None)] * len(tag_list)
#     #             for tag_pos in fock_tag_positions:
#     #                 slice_spec[tag_pos] = -1
#     #             op = op[tuple(slice_spec)]
#     #             tag_list = [tag for tag in tag_list if tag not in fock_tags]

#     #             # Add the tensor and its tags to the list
#     #             tensors_and_tags.append((op, tag_list))

#     #     return tensors_and_tags

#     def __rshift__(self, other: Circuit) -> Circuit:
#         r"connect the two circuits"
#         if not isinstance(other, Circuit):
#             raise TypeError("Can only connect Circuits to Circuits")
#         return Circuit(self.ops + other.ops)

#     def contract(self):
#         opt_einsum_args = [item for pair in self.TN_tensor_list() for item in pair]
#         return opt_einsum.contract(*opt_einsum_args, optimize=settings.OPT_EINSUM_OPTIMIZE)

#     @property
#     def XYd(
#         self,
#     ) -> Tuple[Matrix, Matrix, Vector]:  # NOTE: Overriding Transformation.XYd for efficiency.
#         X = XPMatrix(like_1=True)
#         Y = XPMatrix(like_0=True)
#         d = XPVector()
#         for op in self._ops:
#             opx, opy, opd = op.XYd
#             opX = XPMatrix.from_xxpp(opx, modes=(op.modes, op.modes), like_1=True)
#             opY = XPMatrix.from_xxpp(opy, modes=(op.modes, op.modes), like_0=True)
#             opd = XPVector.from_xxpp(opd, modes=op.modes)
#             if opX.shape is not None and opX.shape[-1] == 1 and len(op.modes) > 1:
#                 opX = opX.clone(len(op.modes), modes=(op.modes, op.modes))
#             if opY.shape is not None and opY.shape[-1] == 1 and len(op.modes) > 1:
#                 opY = opY.clone(len(op.modes), modes=(op.modes, op.modes))
#             if opd.shape is not None and opd.shape[-1] == 1 and len(op.modes) > 1:
#                 opd = opd.clone(len(op.modes), modes=op.modes)
#             X = opX @ X
#             Y = opX @ Y @ opX.T + opY
#             d = opX @ d + opd
#         return X.to_xxpp(), Y.to_xxpp(), d.to_xxpp()

#     @property
#     def is_gaussian(self):
#         """Returns `true` if all operations in the circuit are Gaussian."""
#         return all(op.is_gaussian for op in self._ops)

#     @property
#     def is_unitary(self):
#         """Returns `true` if all operations in the circuit are unitary."""
#         return all(op.is_unitary for op in self._ops)

#     def __len__(self):
#         return len(self._ops)

#     _repr_markdown_ = None

#     def __repr__(self) -> str:
#         """String to display the object on the command line."""
#         return circuit_text(self._ops, decimals=settings.CIRCUIT_DECIMALS)

#     def __str__(self):
#         """String representation of the circuit."""
#         return f"< Circuit | {len(self._ops)} ops | compiled = {self._compiled} >"


class Operation:
    def __init__(
        self,
        op: Union[State, Transformation, Measurement],
        input_modes: list[int],
        output_modes: list[int],
        dual_modes: bool = False,
    ):
        self.op = op
        self.input_modes: list[int] = input_modes
        self.output_modes: list[int] = output_modes
        self.dual_modes: bool = dual_modes


class Circuit:
    def __init__(
        self,
        operations: list[Operation] = [],
        input_modes: list[int] = [],
        output_modes: list[int] = [],
        dual_modes: bool = False,
    ):
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.dual_modes = dual_modes
        self.operations = operations
        self.tag_dispenser = TagDispenser()
        for op in self.operations:
            self.set_up_tags(op)

    def set_up_tags(self, op):
        "assigns a unique tag (int) to each mode of the operation accounting for dual modes"
        if isinstance(op, State):
            self._tags: Dict[str, List[int]] = {
                "out_L": [self.tag_dispenser.get_tag() for _ in range(op.num_modes)],
                "in_L": [],
                "out_R": [
                    self.tag_dispenser.get_tag() for _ in range(self.dual_modes * op.num_modes)
                ],
                "in_R": [],
            }
        elif isinstance(op, Transformation):
            self._tags: Dict[str, List[int]] = {
                "out_L": [self.tag_dispenser.get_tag() for _ in range(op.num_modes)],
                "in_L": [self.tag_dispenser.get_tag() for _ in range(op.num_modes)],
                "out_R": [
                    self.tag_dispenser.get_tag() for _ in range(self.dual_modes * op.num_modes)
                ],
                "in_R": [
                    self.tag_dispenser.get_tag() for _ in range(self.dual_modes * op.num_modes)
                ],
            }
        elif isinstance(op, Measurement):
            self._tags: Dict[str, List[int]] = {
                "out_L": [],
                "in_L": [self.tag_dispenser.get_tag() for _ in range(op.num_modes)],
                "out_R": [],
                "in_R": [
                    self.tag_dispenser.get_tag() for _ in range(self.dual_modes * op.num_modes)
                ],
            }

    def can_connect(self, other):
        intersection = set(self.output_modes).intersection(set(other.input_modes))
        input_overlap = set(self.input_modes).intersection(set(other.input_modes) - intersection)
        output_overlap = (set(self.output_modes) - intersection).intersection(
            set(other.output_modes)
        )
        return len(intersection) > 0 and len(input_overlap) == 0 and len(output_overlap) == 0

    def connect(self, other):
        if not self.can_connect(other):
            raise ValueError("Circuits cannot be connected")
        new_input_modes = self.input_modes + [
            mode for mode in other.input_modes if mode not in self.output_modes
        ]
        new_output_modes = other.output_modes + [
            mode for mode in self.output_modes if mode not in other.input_modes
        ]
        new_contents = self.contents + other.contents
        return Circuit(
            new_contents,
            list(sorted(new_input_modes)),
            list(sorted(new_output_modes)),
            self.dual_modes or other.dual_modes,
        )


# %%
from mrmustard.lab import *
from mrmustard.lab.circuit import Circuit, Operation

op1 = Operation(Coherent(1.0), [], [0], False)
op2 = Operation(Rgate(1.0), [0], [0], False)

circuit = Circuit([op1, op2], [0], [0], False)

print(circuit._tags)
# %%
