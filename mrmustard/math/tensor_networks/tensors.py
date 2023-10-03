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

""" Classes for constructing tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Wire:
    r"""Represents a wire in a tensor network.

    Args:
        id: A numerical identifier for this wire.
        mode: The mode represented by this wire.
        is_input: Whether this wire is an input to a tensor or an output.
        is_ket: Whether this wire is on the ket or on the bra side ??.
        connection_id: A numerical identifier for the contraction involving this wire, or ``None``
            if this wire is not contracted.
        connected_to: The identifier of the tensor connected to this wire, or ``None`` if this wire
            is not connected.

    """
    id: int
    mode: int
    is_input: bool
    is_ket: bool
    contraction_id: int

    def __post_init__(self):
        self._connected_to: int | None = None


@dataclass
class WireGroup:
    r"""A group of wires in a tensor network.

    Args:
        ket: ??
        bra: ??

    """
    ket: dict = field(default_factory=dict)
    bra: dict = field(default_factory=dict)


class Tensor(ABC):
    r"""A tensor in a tensor network.

    Args:
        name (str): The name of this tensor.
        input_legs_ket (List[int]): The indeces labelling the input legs on the bra side.
        output_legs_ket (List[int]): The indeces labelling the output legs on the ket side.
        input_legs_bra (List[int]): The indeces labelling the input legs on the bra side.
        output_legs_bra (List[int]): The indeces labelling the output legs on the ket side.
    """
    _id_counter: int = 0  # to give a unique id to all Tensors and Wires
    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        input_legs_ket: list[int] = [],
        output_legs_ket: list[int] = [],
        input_legs_bra: list[int] = [],
        output_legs_bra: list[int] = [],
    ) -> None:
        # set unique self.id and name
        self.id: int = self._new_id()
        self.name: str = name + "_" + str(self.id)

        # initialize ket and bra wire dicts
        self._in = WireGroup()
        self._out = WireGroup()

        # initialize wires by updating the ket and bra dicts
        for mode in input_legs_ket:
            self._in.ket |= {mode: Wire(self._new_id(), mode, True, True, self._new_id())}
        for mode in output_legs_ket:
            self._out.ket |= {mode: Wire(self._new_id(), mode, False, True, self._new_id())}
        for mode in input_legs_bra:
            self._in.bra |= {mode: Wire(self._new_id(), mode, True, False, self._new_id())}
        for mode in output_legs_bra:
            self._out.bra |= {mode: Wire(self._new_id(), mode, False, False, self._new_id())}

    @property
    def wires(self) -> List[Wire]:
        r"""Returns a list of all wires in this tensor.
        The order is MM default: [ket_out, ket_in, bra_out, bra_in].
        However, minimize reliance on ordering in favour of ids and direction/type.
        """
        return (
            list(self.output.ket.values())
            + list(self.input.ket.values())
            + list(self.output.bra.values())
            + list(self.input.bra.values())
        )

    @property
    def contraction_ids(self) -> list[int]:
        r"""Returns a list of all contraction_ids in this Tensor."""
        return [wire.connection_id for wire in self.wires]

    def wire(self, id: int) -> Wire:
        r"""Returns the wire with the given id."""
        for wire in self.wires:
            if wire.id == id:
                return wire

    def _new_id(self) -> int:
        id = Tensor._id_counter
        Tensor._id_counter += 1
        return id

    @property
    def input(self):
        return self._in

    @property
    def output(self):
        return self._out

    @property
    def modes(self) -> list[int]:
        r"""For backward compatibility. Don't overuse.
        It returns a list of modes for this Tensor, unless it's ambiguous."""
        if self.modes_in == self.modes_out:  # transformation on same modes
            return self.modes_in
        elif len(self.modes_in) == 0:  # state
            return self.modes_out
        elif len(self.modes_out) == 0:  # measurement
            return self.modes_in
        else:
            raise ValueError("modes are ambiguous for this Tensor.")

    @property
    def modes_in(self) -> set[int]:
        "Returns the set of input modes that are used by this Tensor."
        return set(self.input.ket | self.input.bra)

    @property
    def modes_out(self) -> set[int]:
        "Returns the set of output modes that are used by this Tensor."
        return set(self.output.ket | self.output.bra)

    @property
    def all_modes(self) -> set[int]:
        "Returns a set of all the modes spanned by this Tensor."
        return self.modes_out + self.modes_in

    @property
    def adjoint(self) -> AdjointView:
        r"""Returns the adjoint view of this Tensor (with new ids). That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""Returns the dual view of this Tensor (with new ids). That is, in <-> out."""
        return DualView(self)

    @property
    def view(self) -> TensorView:
        r"""Returns a view of this Tensor with new ids."""
        return TensorView(self)

    @property
    @abstractmethod
    def fock(self):
        r""" """


class TensorView(Tensor):
    r"""Base class for Tensor views. It remaps the ids of the original Tensor."""

    def __init__(self, circuit_part):
        self._original = circuit_part  # MM object
        super().__init__(
            self._original.name,
            self._original.output.ket.keys(),
            self._original.input.ket.keys(),
            self._original.output.bra.keys(),
            self._original.input.bra.keys(),
        )

        # note that for the zip we are relying on wire ordering, which is not ideal
        self._id_map = {
            wire.id: wire_orig.id for wire, wire_orig in zip(self.wires, self._original.wires)
        }

    def _original_id(self, id):
        return self._id_map[id]

    def __getattr__(self, attr):
        orig_attr = self._original.__getattribute__(attr)
        if callable(orig_attr):

            def method(*args, **kwargs):
                kwargs["id"] = self._original_id(kwargs["id"])
                return orig_attr(*args, **kwargs)

            return method
        return orig_attr


class DualView(TensorView):
    r"""Dual view of a Tensor. It is used to implement the dual.
    It swaps the input and output wires of a Tensor.
    """

    def __new__(cls, circuit_part: Tensor):
        "makes sure that DualView(DualView(circuit_part)) == TensorView(circuit_part)"
        if isinstance(circuit_part, DualView):
            return circuit_part._original
        return super().__new__(cls)

    @property
    def input(self):
        return self._original.output

    @property
    def output(self):
        return self._original.input

    @property
    def dual(self):
        return self._original.view


class AdjointView(TensorView):
    r"""Adjoint view of a Tensor. It is used to implement the adjoint.
    It swaps the ket and bra wires of a Tensor.
    """

    def __new__(cls, circuit_part: Tensor):
        "makes sure that AdjointView(AdjointView(circuit_part)) == TensorView(circuit_part)"
        if isinstance(circuit_part, AdjointView):
            return circuit_part._original
        return super().__new__(cls)

    @property
    def input(self):  # swaps ket and bra
        return WireGroup(ket=self._original.input.bra, bra=self._original.input.ket)

    @property
    def output(self):  # swaps ket and bra
        return WireGroup(ket=self._original.output.bra, bra=self._original.output.ket)

    @property
    def adjoint(self):
        return self._original.view
