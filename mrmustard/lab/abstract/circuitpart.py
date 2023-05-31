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

from mrmustard import settings
from mrmustard.utils.tagdispenser import TagDispenser

# class Wire:
#     r"""A tag of a CircuitPart. It corresponds to a tag going into or coming out of an
#     object in the circuit picture. A tag corresponds to a single mode and it have one
#     or two components (called L and R) depending on the nature of the gate/state that
#     it's attached to. Wires are single when both ends are attached to pure states and/or to
#     unitary transformations, and they are double otherwise.

#     Arguments:
#         origin (Optional[CircuitPart]): the CircuitPart that the tag originates from
#         end (Optional[CircuitPart]): the CircuitPart that the tag is connected to
#         L (Optional[int]): the left tag of the tag (automatically generated if not specified)
#         R (Optional[int]): the right tag of the tag
#         cutoff (Optional[int]): the Fock cutoff of the tag (determined at runtime if not specified)
#     """

#     def __init__(
#         self,
#         origin: Optional[CircuitPart] = None,
#         end: Optional[CircuitPart] = None,
#         L: Optional[int] = None,
#         R: Optional[int] = None,
#         cutoff: Optional[int] = None,
#     ):
#         self.origin = origin
#         self.end = end
#         self.L = L or TagDispenser().get_tag()
#         self.R = R
#         self.cutoff = cutoff

#     @property
#     def is_connected(self) -> bool:
#         "checks if the tag is connected on both ends"
#         return self.origin is not None and self.end is not None

#     def connect_to(self, end: CircuitPart):
#         "connects the end of the tag to another CircuitPart"
#         if self.end:
#             raise ValueError("Wire already connected.")
#         self.end = end

#     @property
#     def dual_enabled(self):
#         "whether the tag has a dual (R) part."
#         return self.R is not None

#     def enable_dual(self):
#         "enables the dual (R) part of the tag."
#         if not self.R:
#             self.R = TagDispenser().get_tag()

#     def __repr__(self):
#         origin = self.origin.name if self.origin is not None else "..."
#         end = self.end.name if self.end is not None else "..."
#         tags = f"{self.L}," + (f"{self.R}" if self.R is not None else "-")
#         a = "==" if self.dual_enabled else "--"
#         return origin + f" {a}({tags}){a}> " + end


class CircuitPart(ABC):
    r"""CircuitPart supplies functionality for constructing circuits out of connected components.
    A CircuitPart does not know about the physics of the objects it represents: it only knows about
    its place in the circuit and how to connect to other CircuitParts.
    """

    input_tag_at_mode: Dict[int, tuple[int, Optional[int]]]
    output_tag_at_mode: Dict[int, tuple[int, Optional[int]]]
    name: str
    dual_enabled: bool

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the tags throughout this CircuitPart."
        for tag_pair in self.all_tags:
            tag_pair[1] = TagDispenser().get_tag()

    @property  # NOTE: override this property in subclasses if the subclass has internal tags
    def all_tags(self) -> Iterator[tuple[int, Optional[int]]]:
        "Yields all tag pairs of this CircuitPart (output and input)."
        yield from (t for t in self.output_tag_at_mode.values())
        yield from (t for t in self.input_tag_at_mode.values())

    @property
    def modes_in(self) -> set[int]:
        "Returns the set of input modes that are used by this CircuitPart."
        return set(self.input_tag_at_mode.keys())

    @property
    def modes_out(self) -> set[int]:
        "Returns the set of output modes that are used by this CircuitPart."
        return set(self.output_tag_at_mode.keys())

    @property
    def tags_in(self) -> set[int]:
        r"""Returns the set of input tags that are used by this CircuitPart
        in the order [tag_L, tag_R]
        """
        return [t[0] for t in self.input_tag_at_mode.values()] + [
            t[1] for t in self.input_tag_at_mode.values()
        ]

    @property
    def tags_out(self) -> set[int]:
        r"""Returns the set of output tags that are used by this CircuitPart
        in the order [tag_L, tag_R]
        """
        return [t[0] for t in self.output_tag_at_mode.values()] + [
            t[1] for t in self.output_tag_at_mode.values()
        ]

    # @property
    # def connected(self) -> bool:
    #     "Returns True if at least one tag of this CircuitPart is connected to other CircuitParts."
    #     return any(t.is_connected for t in self.all_tags)

    # @property
    # def fully_connected(self) -> bool:
    #     "Returns True if all tags of this CircuitPart are connected to other CircuitParts."
    #     return all(t.is_connected for t in self.all_tags)

    def can_connect_to(self, other: CircuitPart, mode: int) -> tuple[bool, str]:
        r"""Checks whether this CircuitPart can plug its `mode_out=mode` into the `mode_in=mode` of `other`.
        This is the case if the modes exist and if there is no overlap between the output and input modes.
        In other words, if the operations can be plugged as one would in a circuit diagram.

        Arguments:
            other (CircuitPart): the other CircuitPart
            mode (int): the mode to check

        Returns:
            (bool, str): whether the connection is possible and an failure message
        """
        if mode not in self.modes_out:
            return False, f"mode {mode} not an output of {self}."

        if mode not in other.modes_in:
            return False, f"mode {mode} not an input of {other}."

        if self.dual_enabled != other.dual_enabled:
            return False, "dual tags mismatch"

        intersection = self.modes_out.intersection(other.modes_in)
        input_overlap = self.modes_in.intersection(other.modes_in) - intersection
        output_overlap = self.modes_out.intersection(other.modes_out) - intersection

        if len(input_overlap) > 0:
            return False, f"input modes overlap ({input_overlap})"

        if len(output_overlap) > 0:
            return False, f"output modes overlap ({output_overlap})"

        return True, ""

    def connect_to(self, other: CircuitPart, mode: int) -> tuple[int, Optional[int]]:
        "Forward-connect self to other at the given mode. returns the common tag."
        can, fail_reason = self.can_connect_to(other, mode)
        if not can:
            raise ValueError(fail_reason)
        return self.output_tag_at_mode[mode]
