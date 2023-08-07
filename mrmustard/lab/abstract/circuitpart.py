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
from typing import Optional


class CircuitPart:
    r"""CircuitPart supplies functionality for constructing circuits out of components.
    A CircuitPart does not know about its "physical role", rather it is only concerned with the
    connectivity of its own wires. Note that CircuitPart implements only computed properties.

    Effectively it enables any Kraus operator constructed out of tensor network operations.

    A CircuitPart provides the following functionality:
        - it can check whether it can be connected to another CircuitPart at a given mode
        - it can return its input and output modes
        - it keeps a list of its wires as Wire objects
        - keeps a reference to the State/Transformation/Measurement associated with this CircuitPart
    """
    _id_counter: int = 0
    _wire_id_counter: int = 0
    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

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
            modes_output_ket: where the output modes on the ket side exist for this Circuitpart.
            modes_input_ket: where the input modes on the ket side exist for this Circuitpart.
            modes_output_bra: where the output modes on the bra side exist for this Circuitpart.
            modes_input_bra: where the input modes on the bra side exist for this Circuitpart.
        """
        self.id: int = CircuitPart._id_counter
        CircuitPart._id_counter += 1

        self.name: str = name + "_" + str(self.id)

        # note there is no default mechanism for
        # validating modes and ids fields
        self._input = namedtuple("input", ["ket", "bra"])
        self._input.ket = dict()
        self._input.bra = dict()
        self._output = namedtuple("output", ["ket", "bra"])
        self._output.ket = dict()
        self._output.bra = dict()

        for mode in modes_output_ket:
            self._output.ket[mode] = CircuitPart._wire_id_counter
            CircuitPart._wire_id_counter += 1
        for mode in modes_input_ket:
            self._input.ket[mode] = CircuitPart._wire_id_counter
            CircuitPart._wire_id_counter += 1
        for mode in modes_output_bra:
            self._output.bra[mode] = CircuitPart._wire_id_counter
            CircuitPart._wire_id_counter += 1
        for mode in modes_input_bra:
            self._input.bra[mode] = CircuitPart._wire_id_counter
            CircuitPart._wire_id_counter += 1

    @property
    def input(self):  # override in views
        return self._input

    @property
    def output(self):  # override in views
        return self._output

    @property
    def only_ket(self) -> bool:
        return len(self.input.bra) == 0 and len(self.output.bra) == 0

    @property
    def only_bra(self) -> bool:
        return len(self.input.ket) == 0 and len(self.output.ket) == 0

    @property
    def has_ket(self) -> bool:
        return len(self.input.ket) > 0 or len(self.output.ket) > 0

    @property
    def has_bra(self) -> bool:
        return len(self.input.bra) > 0 or len(self.output.bra) > 0

    @property
    def one_sided(self) -> bool:
        return self.only_ket or self.only_bra

    @property
    def two_sided(self) -> bool:
        return not self.one_sided

    @property
    def ids(self) -> list[int]:
        r"""Returns the list of wire ids for this CircuitPart."""
        in_ket = list(self.input.ket.values())
        in_bra = list(self.input.bra.values())
        out_ket = list(self.output.ket.values())
        out_bra = list(self.output.bra.values())
        return in_ket + in_bra + out_ket + out_bra

    @property
    def modes(self) -> list[int]:
        r"""For backward compatibility.
        It returns a list of modes for this CircuitPart, unless it's ambiguous."""
        if self.modes_in == self.modes_out:  # transformation (on same modes)
            return list(self.modes_in)
        elif len(self.modes_in) == 0:  # state
            return list(self.modes_out)
        elif len(self.modes_out) == 0:  # measurement
            return list(self.modes_in)
        else:
            raise ValueError("modes is ambiguous for this CircuitPart. Please use self.modes_dict.")

    @property
    def modes_in(self) -> list[int]:
        "Returns the tuple of input modes that are used by this CircuitPart."
        return list(sorted(list(set(self.input.ket.keys()).union(set(self.input.bra.keys())))))

    @property
    def modes_out(self) -> list[int]:
        "Returns the tuple of output modes that are used by this CircuitPart."
        return list(sorted(list(set(self.output.ket.keys()).union(set(self.output.bra.keys())))))

    @property
    def has_inputs(self) -> bool:
        r"""Returns whether this CircuitPart has input wires."""
        return len(self.modes_in) > 0

    @property
    def has_outputs(self) -> bool:
        r"""Returns whether this CircuitPart has output wires."""
        return len(self.modes_out) > 0

    @property
    def adjoint(self) -> AdjointView:
        r"""Returns the adjoint of this CircuitPart. That is, L <-> R."""
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""Returns the dual of this CircuitPart. That is, in <-> out."""
        return DualView(self)

    # @property
    # def transpose(self, modes: list[int]) -> TransposeView:
    #     r"""Returns the transpose of this CircuitPart. That is, a new mode ordering.
    #     Arguments:
    #         modes: A list of modes in the new order.

    #     Returns:
    #         A TransposeView of this CircuitPart with the modes in the new order.
    #     """
    #     return TransposeView(self, modes)


class CircuitPartView:
    r"""Base class for CircuitPart views. It is used to implement the adjoint and dual views.
    It is not meant to be used directly.
    """

    def __init__(self, cp: CircuitPart) -> None:
        r"""Initializes a CircuitPartView instance, which wraps
        around a CircuitPart and replaces the input and output attributes.

        Arguments:
            cp: The CircuitPart to view.
        """
        self.cp = cp


class DualView(CircuitPartView, CircuitPart):
    r"""Dual view of a CircuitPart. It is used to implement the dual.
    It swaps the input and output wires of a CircuitPart.
    """

    @property
    def input(self):
        return self.cp.output

    @property
    def output(self):
        return self.cp.input

    @property
    def dual(self):
        return self.cp


class AdjointView(CircuitPartView, CircuitPart):
    r"""Adjoint view of a CircuitPart. It is used to implement the adjoint.
    It swaps the ket and bra wires of a CircuitPart.
    """

    @property
    def input(self):
        _input = namedtuple("input", ["ket", "bra"])
        _input.ket = self.cp.input.bra
        _input.bra = self.cp.input.ket
        return _input

    @property
    def output(self):
        _output = namedtuple("output", ["ket", "bra"])
        _output.ket = self.cp.output.bra
        _output.bra = self.cp.output.ket
        return _output

    @property
    def adjoint(self):
        return self.cp


# class TransposeView(CircuitPart):
#     def __init__(
#         self, cp, order: Optional[list[int]] = None, modes: Optional[list[int]] = None
#     ) -> None:
#         r"""Initializes a TransposeView instance. It takes a CircuitPart and a list of modes
#         or a list of mode indices to transpose the CircuitPart to. It is assumed that the
#         modes are a permutation of the modes of the CircuitPart.

#         Arguments:
#             cp: The CircuitPart to transpose.
#             order: A list of mode indices to transpose the CircuitPart to.
#             modes: A list of modes to transpose the CircuitPart to.

#         Returns:
#             A TransposeView of the CircuitPart with the modes in the new order.
#         """
#         if order and modes:
#             raise ValueError("Cannot specify both order and modes.")
#         if not order and not modes:
#             raise ValueError("Must specify either order or modes.")
#         self.order = order or [cp.modes.index(mode) for mode in modes]
#         self.modes = modes or [cp.modes[i] for i in order]
#         super().__init__(cp)

#     @property
#     def input(self):
#         _input = namedtuple("input", ["ket", "bra"])
#         _input.ket = {self.modes[o]: self.cp.input.ket[self.modes[o]] for i,o in self.order}
#             mode: self.cp.input.bra[self.cp.modes[self.order[mode]]]
#             for mode in range(len(self.order))
#         }
#         return _input

#     @property
#     def output(self):
#         _output = namedtuple("output", ["ket", "bra"])
#         return _output
