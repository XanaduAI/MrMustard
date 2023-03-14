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

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Iterator

from mrmustard import settings
from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.utils.circdrawer import circuit_text
from mrmustard.utils.tagdispenser import TagDispenser
from mrmustard.typing import Tensor

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


class Wire:
    r"""A wire of a CircuitPart. It corresponds to a wire going into or coming out of an
    object in the circuit picture. Wires correspond to a single mode.
    Wires are single or double (i.e. having only L or also R component) depending on the nature of the object.
    As a rule of thumb wires are single when states are pure and operations are unitary, and they are double
    when states are density matrices and operations are non-unitary (or unitary but they act on density matrices).

    Arguments:
        origin CircuitPart: the CircuitPart that the wire belongs to
        mode (int): the mode of the wire
        L (int): the left tag of the wire
        R (Optional[int]): the right tag of the wire
        cutoff (Optional[int]): the Fock cutoff of the wire, defaults to settings.WIRE_CUTOFF
        end (Optional[CircuitPart]): the CircuitPart that the wire is connected to
    """

    def __init__(
        self,
        origin: CircuitPart,
        mode: int,
        L: int,
        R: Optional[int] = None,
        cutoff: Optional[int] = settings.WIRE_CUTOFF,
        end: Optional[CircuitPart] = None,
    ):
        self.origin = origin
        self.end = end
        self.mode = mode
        self._L = L
        self._R = R
        self.cutoff = cutoff

    @property
    def L(self):
        "left tag of the wire"
        return self._L

    @L.setter
    def L(self, value):
        if self._L != value:
            TagDispenser().give_back_tag(self._L)
            self._L = value

    @property
    def R(self):
        "right tag of the wire"
        return self._R

    @R.setter
    def R(self, value):
        if self._R != value and self._R is not None:
            TagDispenser().give_back_tag(self._R)
            self._R = value

    @property
    def is_connected(self) -> bool:
        "checks if wire is connected to another operation."
        return self.origin is not None and self.end is not None

    @property
    def double(self):
        "whether the wire has an R component."
        return self.R is not None

    def enable_dual_part(self):
        "enables the dual (R) part of the wire."
        if self.R:
            raise ValueError("Wire already has dual component.")
        self.R = TagDispenser().get_tag()

    def __eq__(self, other: Union[Wire, int]):
        "checks if wires are on the same mode"
        if isinstance(other, Wire):
            return self.mode == other.mode
        elif isinstance(other, int):
            return self.mode == other
        else:
            return False

    def __repr__(self):
        return (
            f"Wire: {self.origin}" + " ==> "
            if self.double
            else " --> " + f"{self.end} | mode={self.mode}, L={self.L}, R={self.R}."
        )

    def __del__(self):
        TagDispenser().give_back_tag(self.L, self.R)


class CircuitPart(ABC):
    r"""CircuitPart supplies functionality for constructing circuits out of connected components."""

    dual_wires_enabled: bool
    input_wires: Dict[int, Wire]
    output_wires: Dict[int, Wire]

    @abstractmethod
    def enable_dual_wires(self) -> None:
        "Enables the dual (R) part of all the wires throughout this CircuitPart."

    @property
    def all_wires(self) -> Iterator[Wire]:
        "Yields all wires of this CircuitPart (input and output)."
        yield from self.input_wires.values()
        yield from self.output_wires.values()

    @property
    def connected(self) -> bool:
        "Returns True if at least one wire of this CircuitPart is connected to other CircuitParts."
        return any(w.is_connected for w in self.all_wires)

    def plugin_modes(self, other: CircuitPart) -> set[int]:
        "Returns the set of modes that would be used if self was plugged into other."
        return set(self.output_wires.keys()).intersection(set(other.input_wires.keys()))

    def can_connect(self, other: CircuitPart) -> tuple[bool, str]:
        r"""Checks whether this Operation can plug into another one at the given mode.
        This is the case if the modes are available, if they are not already connected
        and if there is no overlap between the output and input modes. In other words,
        the operations can be plugged-in as if graphically in a circuit diagram.

        Arguments:
            other (Operation): the other Operation

        Returns:
            bool: whether the connection is possible
        """
        if self.dual_wires_enabled != other.dual_wires_enabled:
            return False, "dual wires mismatch"

        intersection = self.plugin_modes(other)

        if len(intersection) == 0:
            return False, "no common modes"

        for mode in intersection:
            if self.output_wires[mode].end is not None or other.input_wires[mode].end is not None:
                return False, "common mode already connected"

        input_overlap = set(self.input_wires.keys()).intersection(
            set(other.input_wires.keys()) - intersection
        )
        if len(input_overlap) > 0:
            return False, "input modes overlap"

        output_overlap = (set(self.output_wires.keys()) - intersection).intersection(
            set(other.output_wires.keys())
        )
        if len(output_overlap) > 0:
            return False, "output modes overlap"

        return True, ""

    def connect_wires(self, other: CircuitPart) -> None:
        "Forward-connect the given CircuitPart to another CircuitPart."
        can, reason = self.can_connect(other)
        if not can:
            raise ValueError(reason)
        for mode in self.plugin_modes(other):
            self.output_wires[mode].end = other
            other.input_wires[mode].end = self
            other.input_wires[mode].L = self.output_wires[mode].L
            other.input_wires[mode].R = self.output_wires[mode].R


class Operation(CircuitPart):
    r"""A container for States, Transformations and Measurements that allows one to place them
    inside a circuit. It contains information about which modes in the circuit the operation
    is attached to via its wires."""

    def __init__(
        self,
        op: Union[State, Transformation, Measurement],
        input_modes: list[int],
        output_modes: list[int],
        dual_wires_enabled: bool = False,
    ):
        self.op = op
        self.dual_wires_enabled: bool = dual_wires_enabled
        self.input_wires: Dict[int, Wire] = {
            m: Wire(
                origin=self,
                mode=m,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if dual_wires_enabled else None,
            )
            for m in input_modes
        }
        self.output_wires: Dict[int, Wire] = {
            m: Wire(
                origin=self,
                mode=m,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if dual_wires_enabled else None,
            )
            for m in output_modes
        }

        self.num_out = len(output_modes)
        self.num_in = len(input_modes)

    def enable_dual_wires(self) -> None:
        "Enables the dual (R) part of all the wires throughout this Operation."
        for wire in self.all_wires:
            wire.enable_dual_part()

    def __hash__(self):
        "hash function so that Operations can be used as keys in dictionaries."
        tags = tuple(tag for wire in self.all_wires for tag in [wire.L, wire.R] if tag is not None)
        return hash(tags)

    def __repr__(self):
        return (
            f"Operation[{self.op.__class__.__qualname__}](in={list(self.input_wires.keys())},"
            + f"out={list(self.output_wires.keys())}, dual_wires_enabled={self.dual_wires_enabled})"
        )

    def TN_tensor(self) -> Tensor:
        "Returns the TensorNetwork Tensor of this Operation."
        return self.op.TN_tensor()


class Circuit(CircuitPart):
    r"""A collection of interconnected Operations that can be run as a quantum device."""

    def __init__(
        self,
        parts: Optional[list[CircuitPart]] = None,
        dual_wires_enabled: bool = False,
    ):
        self.dual_wires_enabled = dual_wires_enabled
        self.parts = parts or []
        self.connect_all_parts()  # NOTE: to do before setting input and output wires
        self.input_wires: Dict[int, Wire] = {}
        self.output_wires: Dict[int, Wire] = {}
        for part in self.parts:
            for mode, wire in part.input_wires.items():
                if not wire.is_connected:
                    if mode in self.input_wires:
                        raise ValueError("Duplicate input mode.")
                    self.input_wires[mode] = wire
            for mode, wire in part.output_wires.items():
                if not wire.is_connected:
                    if mode in self.output_wires:
                        raise ValueError("Duplicate output mode.")
                    self.output_wires[mode] = wire

    def connect_all_parts(self):
        r"""Connects parts in the circuit according to their input and output modes."""
        for i, part1 in enumerate(self.parts):
            for part2 in self.parts[i + 1 :]:
                if part1.can_connect(part2):
                    part1.connect_wires(part2)
                    if all(wire.is_connected for wire in part1.output_wires.values()):
                        break

    def enable_dual_wires(self) -> None:
        "Enables the dual (R) part of all the wires throughout this Circuit."
        for part in self.parts:
            part.enable_dual_wires()

    def __rshift__(self, other: CircuitPart) -> Circuit:
        other_parts = other.parts if isinstance(other, Circuit) else [other]
        dual = self.dual_wires_enabled or other.dual_wires_enabled
        if dual:
            self.enable_dual_wires()
            other.enable_dual_wires()
        return Circuit(self.parts + other_parts, dual_wires_enabled=dual)

    # a graph representation of the circuit
    # shwoing the connections between the operations
    # def __repr__(self):
    #     import networkx as nx

    #     G = nx.DiGraph()
    #     for op in self.operations:
    #         G.add_node(op)
    #         for mode, wire in op.output_modes.items():
    #             if wire.is_connected:
    #                 G.add_edge(wire.origin, wire.end)
    #     # visualize graph before returning
    #     nx.draw(G)
    #     return nx.nx_pydot.to_pydot(G).to_string()

    _repr_markdown_ = None

    @property
    def ops(self):
        "searches recursively through the circuit and returns a list of operations"
        ops = []
        for part in self.parts:
            if isinstance(part, Operation):
                ops.append(part)
            elif isinstance(part, Circuit):
                ops += part.ops
        return ops

    def __repr__(self) -> str:
        return circuit_text(self.ops, decimals=settings.CIRCUIT_DECIMALS)

    def TN_tensor_list(self) -> list[Tensor]:
        "returns a list of tensors in the tensor network representation of the circuit"
        return [op.TN_tensor for op in self.ops]
