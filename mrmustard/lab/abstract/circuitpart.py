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

from collections import ChainMap, defaultdict, namedtuple

# just type aliases
ID = int
MODE = int


class CircuitPart:
    r"""CircuitPart class for handling modes and wire ids."""
    _id_counter: int = 0  # to give a unique id to all CircuitParts
    _wire_id_counter: int = 0  # to give a unique id to all wires (across CircuitParts too)
    # _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        modes_output_ket: list[MODE] = [],
        modes_input_ket: list[MODE] = [],
        modes_output_bra: list[MODE] = [],
        modes_input_bra: list[MODE] = [],
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
        self.id: ID = CircuitPart._id_counter
        CircuitPart._id_counter += 1
        self.name: str = name + "_" + str(self.id)

        # initialize internal data structures
        self._in = namedtuple("input", ["ket", "bra"])
        self._in.ket: dict[MODE, ID] = dict()
        self._in.bra: dict[MODE, ID] = dict()
        self._out = namedtuple("output", ["ket", "bra"])
        self._out.ket: dict[MODE, ID] = dict()
        self._out.bra: dict[MODE, ID] = dict()

        # to access via self[id] (see __getitem__)

        for mode in modes_output_ket:
            self._out.ket[mode] = self._new_id()
        for mode in modes_input_ket:
            self._in.ket[mode] = self._new_id()
        for mode in modes_output_bra:
            self._out.bra[mode] = self._new_id()
        for mode in modes_input_bra:
            self._in.bra[mode] = self._new_id()

        # self._id_map: dict[ID, ID] = {key: key for key in self.ids}

    # def id_data(self, key: str):
    #     r"""Returns the data dict associated with all wire ids for a given key."""
    #     return {id: data[key] for id, data in self._data.items()}

    def _new_id(self):
        id = CircuitPart._wire_id_counter
        CircuitPart._wire_id_counter += 1
        return id

    def mode_from_id(self, id: ID) -> MODE:
        for mode, id in ChainMap(
            self.input.ket, self.input.bra, self.output.ket, self.output.bra
        ).items():
            if id == id:
                return mode

    @property
    def input(self):
        return self._in

    @property
    def output(self):
        return self._out

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
    def ids(self) -> list[int]:
        r"""Returns the list of wire ids for this CircuitPart."""
        return list(self._data.keys())

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
            raise ValueError(
                "modes are ambiguous for this CircuitPart. Please use self.wire_data['mode']."
            )

    @property
    def modes_in(self) -> list[int]:
        "Returns the list of input modes that are used by this CircuitPart."
        return list(set(self.input.ket | self.input.bra))

    @property
    def modes_out(self) -> list[int]:
        "Returns the list of output modes that are used by this CircuitPart."
        return list(set(self.output.ket | self.output.bra))

    @property
    def all_modes(self) -> set[int]:
        "Returns a list of all the modes spanned by this CircuitPart."
        return list(set(self.modes_out + self.modes_in))

    @property
    def has_inputs(self) -> bool:
        r"""Returns whether this CircuitPart has input wires."""
        return len(self.modes_in) > 0

    @property
    def has_outputs(self) -> bool:
        r"""Returns whether this CircuitPart has output wires."""
        return len(self.modes_out) > 0

    @property
    def only_inputs(self) -> bool:
        return len(self.modes_out) == 0

    @property
    def only_outputs(self) -> bool:
        return len(self.modes_in) == 0

    @property
    def adjoint(self) -> AdjointView:
        r"""Returns the adjoint view of this CircuitPart. That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def dual(self) -> DualView:
        r"""Returns the dual view of this CircuitPart. That is, in <-> out."""
        return DualView(self)


class CircuitPartView:
    r"""Base class for CircuitPart views. It is used to implement the adjoint and dual views.
    It is not meant to be used directly.
    """
    _id_counter: int = 1  # to give a unique id to all CircuitPartViews

    def __init__(self, circuit_part):
        # self.__dict__ = circuit_part.__dict__
        self.original = circuit_part  # MM object
        self.view_id = CircuitPartView._id_counter * 1000_000
        CircuitPartView._id_counter += 1

    def __getattr__(self, attr):
        orig_attr = self.original.__getattribute__(attr)
        if callable(orig_attr):

            def new_func(*args, **kwargs):
                return orig_attr(*args, **kwargs)

            return new_func
        return orig_attr


class DualView(CircuitPartView):
    r"""Dual view of a CircuitPart. It is used to implement the dual.
    It swaps the input and output wires of a CircuitPart.
    """

    def __new__(cls, circuit_part: CircuitPart):
        if isinstance(circuit_part, DualView):
            return circuit_part
        return super().__new__(cls)

    @property
    def input(self):
        return self.original.output  # namedtuple name is still output

    @property
    def output(self):
        return self.original.input  # namedtuple name is still input

    @property
    def dual(self):
        return self.original


class AdjointView(CircuitPartView):
    r"""Adjoint view of a CircuitPart. It is used to implement the adjoint.
    It swaps the ket and bra wires of a CircuitPart.
    """

    def __new__(cls, circuit_part: CircuitPart):
        if isinstance(circuit_part, AdjointView):
            return circuit_part
        return super().__new__(cls)

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
        return self.original


# class UniqueView(CircuitPartView, CircuitPart):
#     r"""A view of a CircuitPart that makes sure that all wires have unique ids."""

#     def __init__(self, original: CircuitPart) -> None:
#         super().__init__(original)  # copy the dict
#         # change all wire ids
#         self._input.ket = {mode: self.view_id + id for mode, id in self.original.input.ket.items()}
#         self._input.bra = {mode: self.view_id + id for mode, id in self.original.input.bra.items()}
#         self._output.ket = {
#             mode: self.view_id + id for mode, id in self.original.output.ket.items()
#         }
#         self._output.bra = {
#             mode: self.view_id + id for mode, id in self.original.output.bra.items()
#         }
#         # change all data ids
#         self._data = {self.view_id + id: val for id, val in self.original._data.items()}


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
