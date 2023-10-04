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

import numpy as np
import uuid


__all__ = ["Wire", "Tensor"]


@dataclass
class Wire:
    r"""Represents a wire in a tensor network.

    Each wire is characterized by a unique identifier ``id``, which must different from
    the identifiers of all the other wires in the tensor network, as well as by a label
    ``mode`` that represents a mode of light.

    Args:
        id: A numerical identifier for this wire.
        mode: The mode represented by this wire.
        is_input: Whether this wire is an input to a tensor or an output.
        is_ket: Whether this wire is on the ket or on the bra side.

    """
    id: int
    mode: int
    is_input: bool
    is_ket: bool

    def __post_init__(self):
        self._contraction_id: int = uuid.uuid1()

    @property
    def contraction_id(self) -> int:
        r"""
        A numerical identifier for the contraction involving this wire, or ``None``
        if this wire is not contracted.
        """
        return self._contraction_id

    @contraction_id.setter
    def contraction_id(self, value):
        self._contraction_id = value


@dataclass
class WireGroup:
    r"""A group of wires in a tensor network.

    Args:
        ket: A dictionary containing the wires on the ket side.
        bra: A dictionary containing the wires on the bra side.

    """
    ket: dict = field(default_factory=dict)
    bra: dict = field(default_factory=dict)


class Tensor(ABC):
    r"""An abstract class representing a tensor in a tensor network.

    In Mr Mustard, tensors are used to represent a state or a transformation on a given set
    of modes in the Fock representation. For example, a single-mode unitary matrix
    :math:`U=\sum_{i,j=1}^Nu_{i,j}|i\rangle\langle{j}|`
    acting on mode ``3`` N-dimensional Fock basis corresponds to the following ``Tensor`` object:

    .. code-block::
    class U(Tensor):
        def value(self, cutoff):
            # specify the value of the tensor
            pass

    U("my_unitary", [3], [3], [3], [3])

    Args:
        name: The name of this tensor.
        input_modes_ket: The input modes on the ket side.
        output_modes_ket: The output modes on the ket side.
        input_modes_bra: The input modes on the bra side.
        output_modes_bra: The output modes on the bra side.
    """
    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        input_modes_ket: list[int] = [],
        output_modes_ket: list[int] = [],
        input_modes_bra: list[int] = [],
        output_modes_bra: list[int] = [],
    ) -> None:
        self._id = uuid.uuid1().int
        self._name = name

        # initialize ket and bra wire dicts
        self._input_wires = WireGroup()
        self._output_wires = WireGroup()

        # initialize wires by updating the ket and bra dicts
        for mode in input_modes_ket:
            self._input_wires.ket |= {mode: Wire(uuid.uuid1().int, mode, True, True)}
        for mode in output_modes_ket:
            self._output_wires.ket |= {mode: Wire(uuid.uuid1().int, mode, False, True)}
        for mode in input_modes_bra:
            self._input_wires.bra |= {mode: Wire(uuid.uuid1().int, mode, True, False)}
        for mode in output_modes_bra:
            self._output_wires.bra |= {mode: Wire(uuid.uuid1().int, mode, False, False)}

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
            return list(self.modes_in)
        elif len(self.modes_in) == 0:  # state
            return list(self.modes_out)
        elif len(self.modes_out) == 0:  # measurement
            return list(self.modes_in)
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

    @abstractmethod
    def value(self, cutoff: int):
        r"""The value of this tensor.

        Args:
            cutoff: the dimension of the Fock basis

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """

    def wire(self, id: int) -> Wire:
        r"""
        The wire with the given ``id``, or ``None`` if no wire corresponds to the given ``id``.
        """
        for wire in self.wires:
            if wire.id == id:
                return wire
        return None


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
