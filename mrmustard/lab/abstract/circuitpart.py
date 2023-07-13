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
from typing import Iterator, Optional, Sequence, Tuple, List, Dict

from mrmustard import settings
from mrmustard.utils.tagdispenser import TagDispenser

Tags = namedtuple("Tags", ["outL", "inL", "outR", "inR"])


class CircuitPart:
    r"""CircuitPart supplies functionality for constructing circuits out of components.
    A CircuitPart does not know about its physical role; it only knows about
    its place in the circuit and how to connect to other CircuitParts.

    Effectively it wraps around Kraus operators constructed out of circuit operations.

    A CircuitPart assigns a pair of integers (L,R) which we call a "tag" to each Fock index.
    In the circuit picture each input and output wire of an element is associated with a tag.

    A CircuitPart provides the following functionality:
        - it can check whether it can be connected to another CircuitPart at a given mode
        - it can be connected to another CircuitPart at a given mode
        - it can assign new tags to its input and output modes (disconnect)
        - it can return its input and output modes
        - it can return all its tags
        - it can return its input and output tags
        - it can return its input and output tags of type L
        - it can return its input and output tags of type R

    Arguments:
        modes_in (list[int]): the input modes of this CircuitPart
        modes_out (list[int]): the output modes of this CircuitPart
        name (str): the name of this CircuitPart
        tag_types (tuple[bool]): tuple of bools indicating which tags to assign (outL, inL, outR, inR)
        **kwargs: additional keyword arguments

    Note:

    """

    def __init__(
        self,
        modes_in: List[int] = [],
        modes_out: List[int] = [],
        name: str = None,
        tag_types: Tuple[bool, ...] = (False,) * 4,
        **kwargs,
    ):
        assert modes_in and (tag_types[1] or tag_types[3])
        assert modes_out and (tag_types[0] or tag_types[2])

        self.tags_out_L: Dict[int, int] = {}
        self.tags_out_R: Dict[int, int] = {}
        self.tags_in_L: Dict[int, int] = {}
        self.tags_in_R: Dict[int, int] = {}

        self.tag_types = tag_types
        self._assign_new_tags(modes_in, modes_out, tag_types)
        self.name = name or self.__class__.__qualname__

        # for k, v in kwargs.items():  # this relies on CircuitPart to be the last in the MRO
        #     setattr(self, k, v)

    _repr_markdown_ = None

    def disconnect(self):
        r"""Disconnects this CircuitPart from other CircuitParts by assigning
        new tags to its input and output modes."""
        self._assign_new_tags(
            self.modes_in,
            self.modes_out,
            self.tag_types,
        )

    def _assign_new_tags(
        self,
        modes_in: Sequence[int],
        modes_out: Sequence[int],
        tag_types: Tuple[bool, ...],
    ):
        r"""Assigns new tags to the input and output modes of this CircuitPart."""
        tags0, tags1, tags2, tags3 = tag_types
        TD = TagDispenser()
        if tags0:
            self.tags_out_L = {m: TD.get_tag() for m in modes_out}
        if tags1:
            self.tags_out_R = {m: TD.get_tag() for m in modes_out}
        if tags2:
            self.tags_in_L = {m: TD.get_tag() for m in modes_in}
        if tags3:
            self.tags_in_R = {m: TD.get_tag() for m in modes_in}

    @property
    def modes(self) -> Optional[list[int]]:
        r"""Returns the modes that this Operation is defined on.
        For backward compatibility, modes fails if modes_in != modes_out."""
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

    @property
    def all_tags(self) -> Iterator[int]:
        "Yields all tags of this CircuitPart (outL, inL, outR, inR)."
        yield from self.tags_out_L.values()
        yield from self.tags_in_L.values()
        yield from self.tags_out_R.values()
        yield from self.tags_in_R.values()

    @property
    def tags_in(self) -> Iterator[int]:
        "Yields all input tags of this CircuitPart (inL, inR)."
        yield from self.tags_in_L.values()
        yield from self.tags_in_R.values()

    @property
    def tags_out(self) -> Iterator[int]:
        "Yields all output tags of this CircuitPart (outL, outR)."
        yield from self.tags_out_L.values()
        yield from self.tags_out_R.values()

    @property
    def modes_in(self) -> tuple[int]:
        "Returns the tuple of input modes that are used by this CircuitPart."
        in_modes = self.tags_in_L.keys().union(self.tags_in_R.keys())
        return tuple(sorted(list(in_modes)))

    @property
    def modes_out(self) -> tuple[int]:
        "Returns the tuple of output modes that are used by this CircuitPart."
        out_modes = self.tags_out_L.keys().union(self.tags_out_R.keys())
        return tuple(sorted(list(out_modes)))

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

        if not set(self.modes_in).isdisjoint(other.modes_in):
            return False, f"input modes overlap {self.modes_in and other.modes_in}"

        if not set(self.modes_out).isdisjoint(other.modes_out):
            return False, f"output modes overlap {self.modes_out and other.modes_out}"

        return True, ""

    def connect_to(self, other: CircuitPart, mode: int, check: bool = False):
        "Forward-connect self to other at the given mode."
        if check:
            can, fail_reason = self.can_connect_to(other, mode)
            if not can:
                raise ValueError(fail_reason)
        other._input_tag_at_mode[mode] = self._output_tag_at_mode[mode]
