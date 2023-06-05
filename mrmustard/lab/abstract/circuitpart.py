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

from collections import namedtuple
from typing import Iterator, Optional, Sequence

from mrmustard import settings
from mrmustard.utils.tagdispenser import TagDispenser

Tag = namedtuple("Tag", ["L", "R"])


class CircuitPart:
    r"""CircuitPart supplies functionality for constructing circuits out of components.
    A CircuitPart does not know about its physical nature, it only knows about
    its place in the circuit and how to connect to other CircuitParts.

    A CircuitPart has tags (pairs of integers) assigned to its input and output modes.
    """

    def __init__(
        self,
        modes_in: list[int] = [],
        modes_out: list[int] = [],
        name: str = None,
    ):
        print("=" * 80)
        print("circuit part init with modes_in", modes_in, "modes_out", modes_out)
        print("self is", self)
        print("=" * 80)
        self._reset_tags(modes_in, modes_out)
        self.name = name or self.__class__.__qualname__

    _repr_markdown_ = None

    def _reset_tags(self, modes_in: Sequence[int] = None, modes_out: Sequence[int] = None):
        r"""Assigns new tags to the input and output modes of this CircuitPart."""
        TD = TagDispenser()
        self.input_tag_at_mode: dict[int, Tag] = {
            m: Tag(TD.get_tag(), TD.get_tag()) for m in modes_in or self.modes_in
        }
        self.output_tag_at_mode: dict[int, tuple[int, Optional[int]]] = {
            m: Tag(TD.get_tag(), TD.get_tag()) for m in modes_out or self.modes_out
        }

    @property
    def modes(self) -> Optional[list[int]]:
        "Returns the modes that this Operation is defined on"
        if self.modes_in == self.modes_out:
            return list(self.modes_in)
        elif len(self.modes_in) == 0:
            return list(self.modes_out)
        elif len(self.modes_out) == 0:
            return list(self.modes_in)
        else:
            raise ValueError(
                "Different input and output modes. Please refer to modes_in and modes_out."
            )

    @property  # NOTE: override this property in subclasses if the subclass has internal tags
    def all_tags(self) -> Iterator[Tag]:
        "Yields all tag pairs of this CircuitPart (output and input)."
        yield from self.output_tag_at_mode.values()
        yield from self.input_tag_at_mode.values()

    @property
    def tags_in(self) -> Iterator[Tag]:
        "Yields all input tag pairs of this CircuitPart."
        yield from self.input_tag_at_mode.values()

    @property
    def tags_out(self) -> Iterator[Tag]:
        "Yields all output tag pairs of this CircuitPart."
        yield from self.output_tag_at_mode.values()

    @property
    def modes_in(self) -> tuple[int]:
        "Returns the tuple of input modes that are used by this CircuitPart."
        return tuple(self.input_tag_at_mode.keys())

    @property
    def modes_out(self) -> tuple[int]:
        "Returns the tuple of output modes that are used by this CircuitPart."
        return tuple(self.output_tag_at_mode.keys())

    @property
    def tags_in_L(self) -> tuple[int]:
        r"""Returns the tuple of input tags of type L that are used by this CircuitPart"""
        return tuple(t.L for t in self.tags_in)

    @property
    def tags_in_R(self) -> tuple[int]:
        r"""Returns the tuple of input tags of type R that are used by this CircuitPart"""
        return tuple(t.R for t in self.tags_out)

    @property
    def tags_out_L(self) -> tuple[int]:
        r"""Returns the tuple of output tags of type L that are used by this CircuitPart"""
        return tuple(t.L for t in self.tags_out)

    @property
    def tags_out_R(self) -> tuple[int]:
        r"""Returns the tuple of output tags of type R that are used by this CircuitPart"""
        return tuple(t.R for t in self.tags_in)

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

    def connect_to(self, other: CircuitPart, mode: int, check: bool = True):
        "Forward-connect self to other at the given mode."
        if check:
            can, fail_reason = self.can_connect_to(other, mode)
            if not can:
                raise ValueError(fail_reason)
        other.input_tag_at_mode[mode] = self.output_tag_at_mode[mode]
