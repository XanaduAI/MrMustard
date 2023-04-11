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

from __future__ import annotations

from abc import ABC
from typing import Dict, Iterator, Optional

from mrmustard.utils.tagdispenser import TagDispenser


class Wire:
    r"""A wire of a CircuitPart. It corresponds to a wire going into or coming out of an
    object in the circuit picture. A wire corresponds to a single mode and it have one
    or two components (called L and R) depending on the nature of the gate/state that
    it's attached to. Wires are single when both ends are attached to pure states and/or to
    unitary transformations, and they are double otherwise.

    Arguments:
        origin (Optional[CircuitPart]): the CircuitPart that the wire originates from
        end (Optional[CircuitPart]): the CircuitPart that the wire is connected to
        L (Optional[int]): the left tag of the wire (automatically generated if not specified)
        R (Optional[int]): the right tag of the wire
        cutoff (Optional[int]): the Fock cutoff of the wire (determined at runtime if not specified)
    """

    def __init__(
        self,
        origin: Optional[CircuitPart] = None,
        end: Optional[CircuitPart] = None,
        L: Optional[int] = None,
        R: Optional[int] = None,
        cutoff: Optional[int] = None,
    ):
        self.origin = origin
        self.end = end
        self.L = L or TagDispenser().get_tag()
        self.R = R
        self.cutoff = cutoff

    @property
    def is_connected(self) -> bool:
        "checks if the wire is connected on both ends"
        return self.origin is not None and self.end is not None

    def connect_to(self, end: CircuitPart):
        "connects the end of the wire to another CircuitPart"
        if self.end:
            raise ValueError("Wire already connected.")
        self.end = end

    @property
    def has_dual(self):
        "whether the wire has a dual (R) part."
        return self.R is not None

    def enable_dual(self):
        "enables the dual (R) part of the wire."
        if self.R:
            raise ValueError("Wire already has dual (R) part.")
        self.R = TagDispenser().get_tag()

    def __repr__(self):
        origin = self.origin.name if self.origin is not None else "..."
        end = self.end.name if self.end is not None else "..."
        tags = f"{self.L}," + (f"{self.R}" if self.R is not None else "-")
        a = "==" if self.has_dual else "--"
        return origin + f" {a}({tags}){a}> " + end


class CircuitPart(ABC):
    r"""CircuitPart supplies functionality for constructing circuits out of connected components.
    A CircuitPart does not know about the physics of the objects it represents: it only knows about
    its place in the circuit and how to connect to other CircuitParts.
    """

    input_wire_at_mode: Dict[int, Wire]
    output_wire_at_mode: Dict[int, Wire]
    name: str
    has_dual: bool

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the wires throughout this CircuitPart."
        for wire in self.all_wires:
            wire.R = TagDispenser().get_tag()
        self.has_dual = True

    @property  # NOTE: override this property in subclasses if the subclass has internal wires
    def all_wires(self) -> Iterator[Wire]:
        "Yields all wires of this CircuitPart (output and input)."
        yield from (w for w in self.output_wire_at_mode.values())
        yield from (w for w in self.input_wire_at_mode.values())

    @property
    def modes_in(self) -> set[int]:
        "Returns the set of input modes that are used by this CircuitPart."
        return set(self.input_wire_at_mode.keys())

    @property
    def modes_out(self) -> set[int]:
        "Returns the set of output modes that are used by this CircuitPart."
        return set(self.output_wire_at_mode.keys())

    @property
    def connected(self) -> bool:
        "Returns True if at least one wire of this CircuitPart is connected to other CircuitParts."
        return any(w.is_connected for w in self.all_wires)

    @property
    def fully_connected(self) -> bool:
        "Returns True if all wires of this CircuitPart are connected to other CircuitParts."
        return all(w.is_connected for w in self.all_wires)

    def can_connect_output_mode(self, other: CircuitPart, mode: int) -> tuple[bool, str]:
        r"""Checks whether this Operation can plug into another one, at a given mode.
        This is the case if the modes exist, if they are not already connected
        and if there is no overlap between the output and input indices.
        In other words, the operations can be plugged as in a circuit diagram.

        Arguments:
            other (Operation): the other Operation
            mode (int): the mode to check

        Returns:
            bool: whether the connection is possible
        """
        if mode not in self.modes_out:
            return False, f"mode {mode} not an output of {self}."

        if mode not in other.modes_in:
            return False, f"mode {mode} not an input of {other}."

        if self.has_dual != other.has_dual:
            return False, "dual wires mismatch"

        if self.output_wire_at_mode[mode].end is not None:
            return (
                False,
                f"output of {self} already connected to {self.output_wire_at_mode[mode].end}",
            )

        if other.input_wire_at_mode[mode].origin is not None:
            return (
                False,
                f"input of {other} already connected to {other.input_wire_at_mode[mode].origin}",
            )

        intersection = self.modes_out.intersection(other.modes_in)
        input_overlap = self.modes_in.intersection(other.modes_in) - intersection
        output_overlap = self.modes_out.intersection(other.modes_out) - intersection

        if len(input_overlap) > 0:
            return False, f"input modes overlap ({input_overlap})"

        if len(output_overlap) > 0:
            return False, f"output modes overlap ({output_overlap})"

        return True, ""

    def connect_output_mode(self, other: CircuitPart, mode: int) -> None:
        "Forward-connect the wire at the given mode to the given CircuitPart."
        can, fail_reason = self.can_connect_output_mode(other, mode)
        if not can:
            raise ValueError(fail_reason)
        self.output_wire_at_mode[mode].end = other
        # when connected the two CircuitParts share the same wire:
        other.input_wire_at_mode[mode] = self.output_wire_at_mode[mode]
