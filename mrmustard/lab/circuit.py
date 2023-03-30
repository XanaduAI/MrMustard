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

from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Iterator, Optional, Union

from mrmustard import settings
from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.typing import Tensor
from mrmustard.utils.circdrawer import circuit_text
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

#     #             fock_tag_indices = [tag_list.index(tag) for tag in fock_tags if tag in tag_list]
#     #             slice_spec = [slice(None)] * len(tag_list)
#     #             for tag_pos in fock_tag_indices:
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
    object in the circuit picture. Although wire corresponds to a single mode, wires can have one
    or two components (called L and R) depending on the nature of the gate/state.
    Usually wires are single when states are pure and operations are unitary, and they are double
    when states are density matrices and operations are non-unitary (or unitary but they act on density matrices).

    Arguments:
        origin CircuitPart: the CircuitPart that the wire belongs to
        end (Optional[CircuitPart]): the CircuitPart that the wire is connected to
        L (int): the left tag of the wire
        R (Optional[int]): the right tag of the wire
        cutoff (Optional[int]): the Fock cutoff of the wire, defaults to settings.WIRE_CUTOFF
    """

    def __init__(
        self,
        index: int,  # horizontal wires
        origin: Optional[CircuitPart] = None,
        end: Optional[CircuitPart] = None,
        L: Optional[int] = None,
        R: Optional[int] = None,
        cutoff: Optional[int] = settings.WIRE_CUTOFF,
    ):
        self.index = index
        self.origin = origin
        self.end = end
        self.L = L or TagDispenser().get_tag()
        self.R = R
        self.cutoff = cutoff

    @property
    def is_connected(self) -> bool:
        "checks if wire is connected to another operation."
        return self.origin is not None and self.end is not None

    def connect_to(self, end: CircuitPart):
        "connects the wire to another CircuitPart"
        if self.end:
            raise ValueError("Wire already connected.")
        self.end = end

    @property
    def has_dual(self):
        "whether the wire has an R part."
        return self.R is not None

    def enable_dual(self):
        "enables the dual (R) part of the wire."
        if self.R:
            raise ValueError("Wire already has dual (R) part.")
        self.R = TagDispenser().get_tag()

    def __repr__(self):
        origin = self.origin.name if self.origin is not None else "..."
        end = self.end.name if self.end is not None else "..."
        arrow = " ==> " if self.has_dual else " --> "
        wire = f"{origin}[{self.index}] {arrow} [{self.index}]{end}"

        return f"Wire[L={self.L}, R={self.R}]: {wire}"


class CircuitPart(ABC):
    r"""CircuitPart supplies functionality for constructing circuits out of connected components.
    A CircuitPart does not know about the physics of the objects it represents, it only knows about
    the wires that connect to it. It is up to the CircuitPart to decide how to use the wires.

    A CircuitPart has input wires and output wires, attached to indices. Indices are circuit-level
    and are not necessarily the same as mode indices. A CircuitPart has an index_for_mode mapping
    to keep track of which mode is connected to which index. Modes are counted from 0 up to
    the number of modes in the state or transformation, while indices can be any integer. Indices
    are just integer names for the wires in a circuit.
    """

    has_dual: bool
    input_wire_at_index: Dict[int, Wire]
    output_wire_at_index: Dict[int, Wire]
    name: str
    index_for_mode: Dict[int, int]

    @abstractmethod
    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the wires throughout this CircuitPart."

    @property  # NOTE: override this property in subclasses if the subclass has internal wires
    def all_wires(self) -> Iterator[Wire]:
        "Yields all wires of this CircuitPart (output and input)."
        yield from (w for w in self.output_wire_at_index.values())
        yield from (w for w in self.input_wire_at_index.values())

    @property
    def input_indices(self) -> set[int]:
        "Returns the set of input indices that are used by this CircuitPart."
        return set(self.input_wire_at_index.keys())

    @property
    def output_indices(self) -> set[int]:
        "Returns the set of output indices that are used by this CircuitPart."
        return set(self.output_wire_at_index.keys())

    @property
    def connected(self) -> bool:
        "Returns True if at least one wire of this CircuitPart is connected to other CircuitParts."
        return any(w.is_connected for w in self.all_wires)

    def can_connect_index(self, other: CircuitPart, index: int) -> tuple[bool, str]:
        r"""Checks whether this Operation can plug into another one, at a given index.
        This is the case if the indices are available, if they are not already connected
        and if there is no overlap between the output and input indices. In other words,
        the operations can be plugged-in as if graphically in a circuit diagram.

        Arguments:
            other (Operation): the other Operation
            index (int): the index to check

        Returns:
            bool: whether the connection is possible
        """
        if index not in self.output_indices.intersection(other.input_indices):
            return False, f"index {index} not available when connecting {self.name} to {other.name}"

        if self.has_dual != other.has_dual:
            return False, "dual wires mismatch"

        intersection = self.output_indices.intersection(other.input_indices)

        if len(intersection) == 0:
            return False, "no common indices"

        if self.output_wire_at_index[index].end is not None:
            return False, f"output wire already connected to {self.output_wire_at_index[index].end}"

        if other.input_wire_at_index[index].origin is not None:
            return (
                False,
                f"input wire already connected to {other.input_wire_at_index[index].origin}",
            )

        input_overlap = self.input_indices.intersection(other.input_indices) - intersection
        if len(input_overlap) > 0:
            return False, "input indices overlap"

        output_overlap = self.output_indices.intersection(other.output_indices) - intersection
        if len(output_overlap) > 0:
            return False, "output indices overlap"

        return True, ""

    def connect_part_to(self, other: CircuitPart) -> None:
        "Forward-connect the given CircuitPart to another CircuitPart."
        for index in self.output_indices.intersection(other.input_indices):
            self.connect_wire_to(other, index)

    def connect_wire_to(self, other: CircuitPart, index: int) -> None:
        can, fail_reason = self.can_connect_index(other, index)
        if not can:
            raise ValueError(fail_reason)
        self.output_wire_at_index[index].end = other
        other.input_wire_at_index[index] = self.output_wire_at_index[index]


class Operation(CircuitPart):
    r"""A container for States, Transformations and Measurements that allows one to place them
    inside a circuit. It contains information about which indices in the circuit the operation
    is attached to via its wires. The Operation is an abstraction above the physics of the object
    that it contains. Its main purpose is to allow the user to easily construct circuits."""

    def __init__(
        self,
        wrapped: Union[State, Transformation, Measurement],
        input_indices: list[int],
        output_indices: list[int],
        has_dual: bool = False,
    ):
        self.wrapped = wrapped
        self.name = wrapped.__class__.__qualname__
        self.has_dual: bool = has_dual
        self.input_wire_at_index: Dict[int, Wire] = {
            i: Wire(
                index=i,
                end=self,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if has_dual else None,
            )
            for i in input_indices
        }
        self.output_wire_at_index: Dict[int, Wire] = {
            i: Wire(
                index=i,
                origin=self,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if has_dual else None,
            )
            for i in output_indices
        }

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.wrapped, name)

    _repr_markdown_ = None

    def disconnect(self) -> Operation:
        "Re-issue new wires for this operation"
        for wire_dict in [self.input_wire_at_index, self.output_wire_at_index]:
            for i, wire in wire_dict.items():
                wire_dict[i] = Wire(
                    index=i,
                    origin=None if wire.origin is not self else self,
                    end=None if wire.end is not self else self,
                    L=TagDispenser().get_tag(),
                    R=TagDispenser().get_tag() if self.has_dual else None,
                )
        return self

    @property
    def indices(self) -> Optional[list[int]]:
        if self.input_indices == self.output_indices:
            return list(self.input_indices)

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the wires throughout this Operation."
        for wire in self.all_wires:
            wire.enable_dual()

    def __hash__(self):
        "hash function so that Operations can be used as keys in dictionaries."
        tags = tuple(tag for wire in self.all_wires for tag in [wire.L, wire.R] if tag is not None)
        return hash(tags)

    def __repr__(self):
        return (
            f"Operation[{self.wrapped.__class__.__qualname__}](in={list(self.input_indices)}, "
            + f"out={list(self.output_indices)}, has_dual={self.has_dual})"
        )

    def __rshift__(self, other: CircuitPart) -> Circuit:
        other_parts = other.parts if isinstance(other, Circuit) else [other]
        dual = self.has_dual or other.has_dual
        if dual:
            self.enable_dual()
            other.enable_dual()
        return Circuit([self] + other_parts, has_dual=dual)

    def TN_tensor(self) -> Tensor:
        "Returns the TensorNetwork Tensor of this Operation."
        return self.wrapped.TN_tensor()


class Circuit(CircuitPart):
    r"""A collection of interconnected Operations that can be run as a quantum device."""

    def __init__(
        self,
        parts: Optional[list[CircuitPart]] = None,
        has_dual: bool = False,
    ):
        self.has_dual = has_dual
        self.parts = parts or []
        self.connect_all_parts()  # NOTE: to do before setting input and output wires
        self.input_wire_at_index: Dict[int, Wire] = {}
        self.output_wire_at_index: Dict[int, Wire] = {}
        for part in self.parts:
            for index, wire in part.input_wire_at_index.items():
                if not wire.is_connected:
                    if index in self.input_indices:
                        raise ValueError("Duplicate input index.")
                    self.input_wire_at_index[index] = wire
            for index, wire in part.output_wire_at_index.items():
                if not wire.is_connected:
                    if index in self.output_indices:
                        raise ValueError("Duplicate output index.")
                    self.output_wire_at_index[index] = wire
        self.name = "Circuit"

    def connect_all_parts(self):
        r"""Connects parts in the circuit according to their input and output indices."""
        for i, part1 in enumerate(self.parts):
            for index in part1.output_indices:
                for part2 in self.parts[i + 1 :]:
                    if part1.can_connect_index(part2, index)[0]:
                        part1.connect_wire_to(part2, index)
                        break

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the wires throughout this Circuit."
        for part in self.parts:
            part.enable_dual()

    def __rshift__(self, other: CircuitPart) -> Circuit:
        if dual := (self.has_dual or other.has_dual):
            self.enable_dual()
            other.enable_dual()
        return Circuit([self, other], has_dual=dual)

    def flatten(self) -> Circuit:
        "Flattens the circuit."
        return Circuit([op.disconnect() for op in self.ops], has_dual=self.has_dual)

    # def tag_map(self) -> dict[int, int]:
    #     tags = []
    #     for op in self.ops:
    #         for wire in op.all_wires:
    #             tags.append(wire.L)
    #             if wire.R is not None:
    #                 tags.append(wire.R)
    #     # re-issue unique tags starting from 0
    #     tag_map = {}
    #     index = 0
    #     for tag in sorted(tags):
    #         if tag not in tag_map:
    #             tag_map[tag] = index
    #             index += 1
    #     return tag_map

    # _repr_markdown_ = None

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
