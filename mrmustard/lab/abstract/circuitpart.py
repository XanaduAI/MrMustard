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

""" CircuitPart class for constructing circuits out of components."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Wire:
    id: int
    mode: int
    direction: str
    type: str
    contraction_id: int
    owner: CircuitPart
    dimension: Optional[int] = None
    connected_to: Optional[Wire] = None


@dataclass
class WireGroup:
    ket: dict = field(default_factory=dict)
    bra: dict = field(default_factory=dict)


class CircuitPart:
    r"""CircuitPart class for handling modes and wire ids."""
    _id_counter: int = 0  # to give a unique id to all CircuitParts and Wires
    # _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        modes_output_ket: list[int] = [],
        modes_input_ket: list[int] = [],
        modes_output_bra: list[int] = [],
        modes_input_bra: list[int] = [],
    ):
        r"""
        Initializes a CircuitPart instance.

        Arguments:
            name: A string name for this circuit part.
            modes_output_ket: the output modes on the ket side
            modes_input_ket: the input modes on the ket side
            modes_output_bra: the output modes on the bra side
            modes_input_bra: the input modes on the bra side
        """
        # set unique self.id and name
        self.id: int = CircuitPart._id_counter
        CircuitPart._id_counter += 1
        self.name: str = name + "_" + str(self.id)

        # initialize ket and bra wire dicts
        self._in = WireGroup()
        self._out = WireGroup()

        # initialize wires by updating the ket and bra dicts
        for mode in modes_output_ket:
            self._out.ket |= {mode: Wire(self._new_id(), mode, "out", "ket", self._new_id(), self)}
        for mode in modes_input_ket:
            self._in.ket |= {mode: Wire(self._new_id(), mode, "in", "ket", self._new_id(), self)}
        for mode in modes_output_bra:
            self._out.bra |= {mode: Wire(self._new_id(), mode, "out", "bra", self._new_id(), self)}
        for mode in modes_input_bra:
            self._in.bra |= {mode: Wire(self._new_id(), mode, "in", "bra", self._new_id(), self)}

    @property
    def wires(self):
        r"""Returns a list of all wires in this CircuitPart.
        The order is MM default: [ket_out, ket_in, bra_out, bra_in].
        """
        return (
            list(self.output.ket.values())
            + list(self.input.ket.values())
            + list(self.output.bra.values())
            + list(self.input.bra.values())
        )

    @property
    def contraction_ids(self):
        r"""Returns a list of all contraction_ids in this CircuitPart."""
        return [wire.contraction_id for wire in self.wires]

    def wire(self, id: int) -> Wire:
        r"""Returns the wire with the given id."""
        for wire in self.wires:
            if wire.id == id:
                return wire

    def _new_id(self):
        id = CircuitPart._id_counter
        CircuitPart._id_counter += 1
        return id

    @property
    def input(self):
        return self._in

    @property
    def output(self):
        return self._out

    # @property
    # def has_only_ket(self) -> bool:
    #     return len(self.input.bra) == 0 and len(self.output.bra) == 0

    # @property
    # def has_only_bra(self) -> bool:
    #     return len(self.input.ket) == 0 and len(self.output.ket) == 0

    # @property
    # def has_ket(self) -> bool:
    #     return len(self.input.ket) > 0 or len(self.output.ket) > 0

    # @property
    # def has_bra(self) -> bool:
    #     return len(self.input.bra) > 0 or len(self.output.bra) > 0

    @property
    def ids(self) -> list[int]:
        r"""Returns the list of wire ids for this CircuitPart."""
        return [wire.id for wire in self.wires]

    @property
    def modes(self) -> list[int]:
        r"""For backward compatibility.
        It returns a list of modes for this CircuitPart, unless it's ambiguous."""
        if self.modes_in == self.modes_out:  # transformation on same modes
            return self.modes_in
        elif len(self.modes_in) == 0:  # state
            return self.modes_out
        elif len(self.modes_out) == 0:  # measurement
            return self.modes_in
        else:
            raise ValueError("modes are ambiguous for this CircuitPart.")

    @property
    def modes_in(self) -> list[int]:
        "Returns the set of input modes that are used by this CircuitPart."
        return list(self.input.ket | self.input.bra)

    @property
    def modes_out(self) -> set[int]:
        "Returns the set of output modes that are used by this CircuitPart."
        return list(self.output.ket | self.output.bra)

    @property
    def all_modes(self) -> set[int]:
        "Returns a list of all the modes spanned by this CircuitPart."
        return self.modes_out + self.modes_in

    # @property
    # def has_inputs(self) -> bool:
    #     r"""Returns whether this CircuitPart has input wires."""
    #     return len(self.modes_in) > 0

    # @property
    # def has_outputs(self) -> bool:
    #     r"""Returns whether this CircuitPart has output wires."""
    #     return len(self.modes_out) > 0

    # @property
    # def has_only_inputs(self) -> bool:
    #     return len(self.modes_out) == 0

    # @property
    # def has_only_outputs(self) -> bool:
    #     return len(self.modes_in) == 0

    @property
    def adjoint(self) -> AdjointView:
        r"""Returns the adjoint view of this CircuitPart (with new ids). That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""Returns the dual view of this CircuitPart (with new ids). That is, in <-> out."""
        return DualView(self)

    @property
    def view(self) -> CircuitPartView:
        r"""Returns a view of this CircuitPart with new ids."""
        return CircuitPartView(self)


class CircuitPartView(CircuitPart):
    r"""Base class for CircuitPart views. It remaps the ids of the original CircuitPart."""

    def __init__(
        self, circuit_part, modes_output_ket, modes_input_ket, modes_output_bra, modes_input_bra
    ):
        self.original = circuit_part  # MM object
        super().__init__(
            self.original.name,
            modes_output_ket or self.original.modes_output_ket,
            modes_input_ket or self.original.modes_input_ket,
            modes_output_bra or self.original.modes_output_bra,
            modes_input_bra or self.original.modes_input_bra,
        )

        self._id_map = {id: orig_id for id, orig_id in zip(self.ids, self.original.ids)}

    def original_id(self, id):
        return self._id_map[id]

    def __getattr__(self, attr):
        orig_attr = self.original.__getattribute__(attr)
        if callable(orig_attr):

            def method(*args, **kwargs):
                return orig_attr(*args, **kwargs)

            return method
        return orig_attr


class DualView(CircuitPartView):
    r"""Dual view of a CircuitPart. It is used to implement the dual.
    It swaps the input and output wires of a CircuitPart.
    """

    # def __new__(cls, circuit_part: CircuitPart):
    #     "makes sure that DualView(DualView(circuit_part)) == CircuitPartView(circuit_part)"
    #     if isinstance(circuit_part, DualView):
    #         return circuit_part.view
    #     return super().__new__(cls)

    @property
    def input(self):
        return self.original.output  # namedtuple name is still output

    @property
    def output(self):
        return self.original.input  # namedtuple name is still input

    @property
    def dual(self):
        return self.original.view


class AdjointView(CircuitPartView):
    r"""Adjoint view of a CircuitPart. It is used to implement the adjoint.
    It swaps the ket and bra wires of a CircuitPart.
    """

    # def __new__(cls, circuit_part: CircuitPart):
    #     if isinstance(circuit_part, AdjointView):
    #         return circuit_part.view
    #     return super().__new__(cls)

    @property
    def input(self):
        # swaps ket and bra
        _input = namedtuple("input", ["ket", "bra"])
        _input.ket = self.original.input.bra
        _input.bra = self.original.input.ket
        return _input

    @property
    def output(self):
        # swaps ket and bra
        _output = namedtuple("output", ["ket", "bra"])
        _output.ket = self.original.output.bra
        _output.bra = self.original.output.ket
        return _output

    @property
    def adjoint(self):
        return self.original.view
