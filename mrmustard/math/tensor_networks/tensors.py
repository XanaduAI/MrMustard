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
from typing import Callable, List

import numpy as np


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
        input_wires_ket (List[int]): The indeces labelling the input wires on the ket side.
        output_wires_ket (List[int]): The indeces labelling the output wires on the ket side.
        input_wires_bra (List[int]): The indeces labelling the input wires on the bra side.
        output_wires_bra (List[int]): The indeces labelling the output wires on the bra side.
    """
    _id_counter: int = 0  # to give a unique id to all Tensors and Wires
    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        input_wires_ket: list[int] = [],
        output_wires_ket: list[int] = [],
        input_wires_bra: list[int] = [],
        output_wires_bra: list[int] = [],
    ) -> None:
        self._id = self._new_id()
        self._name = name

        # initialize ket and bra wire dicts
        self._input_wires = WireGroup()
        self._output_wires = WireGroup()

        # initialize wires by updating the ket and bra dicts
        for mode in input_wires_ket:
            self._input_wires.ket |= {mode: Wire(self._new_id(), mode, True, True, self._new_id())}
        for mode in output_wires_ket:
            self._output_wires.ket |= {
                mode: Wire(self._new_id(), mode, False, True, self._new_id())
            }
        for mode in input_wires_bra:
            self._input_wires.bra |= {mode: Wire(self._new_id(), mode, True, False, self._new_id())}
        for mode in output_wires_bra:
            self._output_wires.bra |= {
                mode: Wire(self._new_id(), mode, False, False, self._new_id())
            }

    @property
    def adjoint(self) -> AdjointView:
        r"""Returns the adjoint view of this Tensor (with new ``id``s). That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def id(self) -> int:
        r"""
        The unique identifier of this tensor.
        """
        return self._id

    @property
    def name(self) -> int:
        r"""
        The name of this tensor.
        """
        return self._name

    @property
    def wires(self) -> List[Wire]:
        r"""
        The list of all wires in this tensor.
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
        r"""
        The list of all contraction_ids in this Tensor."""
        return [wire.connection_id for wire in self.wires]

    def wire(self, id: int) -> Wire:
        r"""
        The wire with the given ``id``, or ``None`` if no wire corresponds to the given ``id``.
        """
        for wire in self.wires:
            if wire.id == id:
                return wire
        return None

    def _new_id(self) -> int:
        id = Tensor._id_counter
        Tensor._id_counter += 1
        return id

    @property
    def input(self):
        return self._input_wires

    @property
    def output(self):
        return self._output_wires

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
    @abstractmethod
    def value(self):
        r"""The value of this tensor."""


class TensorView(Tensor):
    r"""
    Base class for tensor views. It remaps the ids of the original Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            self._original.name,
            self._original.input.ket.keys(),
            self._original.output.ket.keys(),
            self._original.input.bra.keys(),
            self._original.output.bra.keys(),
        )

    @property
    def value(self):
        r""" """
        return self._original.value


class AdjointView(Tensor):
    r"""
    Adjoint view of a tensor. It swaps the ket and bra wires of a Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            self._original.name,
            self._original.input.bra.keys(),
            self._original.output.bra.keys(),
            self._original.input.ket.keys(),
            self._original.output.ket.keys(),
        )

    @property
    def value(self):
        r""" """
        return np.conj(self._original.value).T
