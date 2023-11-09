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
from typing import List, Optional, Tuple, Iterable
from mrmustard.typing import ComplexVector

import uuid

from mrmustard.math import Math

math = Math()


__all__ = ["Wire", "Tensor"]


def random_int() -> int:
    r"""
    Returns a random integer obtained from a UUID
    """
    return uuid.uuid1().int



# class CircuitAPI:
    # def __init__(
    #     self,
    #     name: str,
    #     modes_in_ket: Optional[list[int]] = None,
    #     modes_out_ket: Optional[list[int]] = None,
    #     modes_in_bra: Optional[list[int]] = None,
    #     modes_out_bra: Optional[list[int]] = None,
    # ) -> None:
        
    #     msg = "modes on ket and bra sides must be equal, unless either of them is `None`."
    #     if modes_in_ket and modes_in_bra and modes_in_ket != modes_in_bra:
    #         raise ValueError(f"Input {msg}")
    #     if modes_out_ket and modes_out_bra and modes_out_ket != modes_out_bra:
    #         raise ValueError(f"Output {msg}")

    #     self._name = name

    #     # initialize ket and bra wire dicts
    #     self._in_ket = {m: Wire(random_int()) for m in modes_in_ket}
    #     self._in_bra = {m: Wire(random_int()) for m in modes_in_bra}
    #     self._out_ket = {m: Wire(random_int()) for m in modes_out_ket}
    #     self._out_bra = {m: Wire(random_int()) for m in modes_out_bra}

    # @property
    # def input(self):
    #     r"""
    #     A mapping the input modes to their respective wires.
    #     """
    #     return WiresKetBra(self._in_ket, self._in_bra)
    
    # @property
    # def output(self):
    #     r"""
    #     A mapping the output modes to their respective wires.
    #     """
    #     return WiresKetBra(self._out_ket, self._out_bra)

    # def _flag(self, modes: int | Iterable[int]):
    #     modes = [modes] if isinstance(modes, int) else modes
    #     new = self.__class__.__new__(self.__class__)
    #     new.__dict__ = self.__dict__.copy()
    #     for m in self.modes:
    #         if m in new._in_ket:
    #             new._in_ket[m].flagged = m in modes
    #         if m in new._in_bra:
    #             new._in_bra[m].flagged = m in modes
    #         if m in new._out_ket:
    #             new._out_ket[m].flagged = m in modes
    #         if m in new._out_bra:
    #             new._out_bra[m].flagged = m in modes
    #     return new
        
    # @property
    # def input(self):
    #     return self._flag(self.modes_in)
    
    # @property
    # def output(self):
    #     return self._flag(self.modes_out)
    
    # def __getitem__(self, modes):
    #     return self._flag(modes)





@dataclass
class Wire:
    r"""Represents a wire in a tensor network.

    Each wire is characterized by a unique identifier ``id``, which must be different from
    the identifiers of all the other wires in the tensor network.

    Args:
        id: A numerical identifier for this wire.
        flagged: Whether this wire is flagged or not.
        index: The index of this wire in the tensor indices of the underlying tensor.

    """
    # id: int
    flagged: bool = False
    index: int = None

    def __post_init__(self):
        self.contraction_id: int = random_int()
        self.dim = None  # CV is None, DV is an int
        self.is_connected = False

@dataclass
class WireGroup:
    r"""A single group of wires.
    It's essentially a dict that can also do
    `wg[1,2,3]` and return `{1: wg[1], 2: wg[2], 3: wg[3]}`.

    Also, set(wg) is the set of its modes (ints).
    This allows for easy comparison of WireGroups without having to compare the wires.

    Args:
        wires: A dictionary containing the wires in this group.
    """
    wires: dict[int, Wire] = field(default_factory=dict)

    def __getitem__(self, modes: int | Iterable[int]) -> Wire | WireGroup:
        r"""If ``modes`` is an integer, it returns the wire at that mode.
        If ``modes`` is an iterable, it returns a new WireGroup containing the wires at those modes.
        """
        if isinstance(modes, int):
            return self.wires[modes]  # TODO: fix: not a WireGroup
        return WireGroup({m: self.wires[m] for m in modes if m in self.wires})

    def __setitem__(self, mode: int, wire: Wire):
        self.wires[mode] = wire

    def __hash__(self):
        return hash(self.wires.keys())

    def __eq__(self, other):
        return self.wires.keys() == other.wires.keys()

    def keys(self):
        return self.wires.keys()

    def values(self):
        return self.wires.values()


@dataclass
class WiresKetBra:
    r"""A pair of WireGroups representing ket and bra sides.

    Args:
        ket (WireGroup): The ket side.
        bra (WireGroup): The bra side.

    """
    ket: WireGroup = field(default_factory=WireGroup)
    bra: WireGroup = field(default_factory=WireGroup)

    def __getitem__(self, modes: int | Iterable[int]) -> WiresKetBra:
        r"""Enables accessing wires with the syntax e.g. `self.input[4].ket` and it allows creating a
        WiresKetBra with ket and bra at a mode: e.g. `self.output[3,4,5]` without specifying ket/bra.
        """
        modes = [modes] if isinstance(modes, int) else modes
        if len(self.ket) > 0:
            ket = {m: self.ket[m] for m in modes}
        if len(self.bra) > 0:
            bra = {m: self.bra[m] for m in modes}
        return WiresKetBra(ket, bra)

    # def __setitem__(self, mode: int, wire: Wire):
    #     try:
    #         if wire.is_ket:
    #             side = "ket"
    #             self.ket[mode] = wire
    #         else:
    #             side = "bra"
    #             self.bra[mode] = wire
    #     except KeyError:
    #         raise ValueError(f"Cannot set wire at mode {mode} on {side} side")


class TensorAPI:
    r"""A class exposing the Tensor Network API.
    Tensors in a tensor network should inherit from this class.

    The TensorAPI manages the wires of a tensor while keeping track of
    the order of the indices of the underlying tensor.

    We distinguish between input and output wires, and between ket and bra sides.

    Args:
        name: The name of this tensor.
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
    """
    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ) -> None:
        self.name = name
        self._update_modes(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    def _rewire(
        self,
        modes_out_bra: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_ket: Optional[list[int]] = None,
    ) -> None:
        r"""
        Updates the modes in this tensor by setting:

          * self._modes_out_bra, a list of output modes on the bra side
          * self._modes_in_bra, a list of input modes on the bra side
          * self._modes_out_ket, a list of output modes on the ket side
          * self._modes_in_ket, a list of input modes on the ket side
          * self.self._input, a WiresKetBra containing all the input modes
          * self.self._output, a WiresKetBra containing all the output modes

        Note that it creates brand new wires.
        If you want to rewire a tensor without creating new wires,
        use ``change_modes`` instead.

        Raises:
            ValueError: if `modes_in_ket` and `modes_in_bra` are not equal, and neither
                of them is `None`.
            ValueError: if `modes_out_ket` and `modes_out_bra` are not equal, and neither
                of them is `None`.
        """
        msg = "modes on ket and bra sides must be equal, unless either of them is `None`."
        if modes_in_ket and modes_in_bra:
            if modes_in_ket != modes_in_bra:
                msg = f"Input {msg}"
                raise ValueError(msg)
        if modes_out_ket and modes_out_bra:
            if modes_out_ket != modes_out_bra:
                msg = f"Output {msg}"
                raise ValueError(msg)

        self._modes_out_bra = modes_out_bra if modes_out_bra else []
        self._modes_in_bra = modes_in_bra if modes_in_bra else []
        self._modes_out_ket = modes_out_ket if modes_out_ket else []
        self._modes_in_ket = modes_in_ket if modes_in_ket else []

        # initialize ket and bra wire dicts
        ket = {m: Wire(False, i) for i,m in enumerate(self._modes_in_ket)}
        IK = len(self._modes_in_ket)
        bra = {m: Wire(False, i) for i,m in enumerate(self._modes_in_bra)+IK}
        IB = len(self._modes_in_bra)
        self._input = WiresKetBra(ket, bra)

        ket = {m: Wire(False, i) for i,m in enumerate(self._modes_out_ket)+IK+IB}
        OK = len(self._modes_out_ket)
        bra = {m: Wire(False, i) for i,m in enumerate(self._modes_out_bra)+IK+IB+OK}
        self._output = WiresKetBra(ket, bra)

    @property
    def input(self):
        r"""
        Returns a new tensor with input wires flagged and output wires unflagged.
        The new tensor's __dict__ is a copy of this tensor's __dict__.
        """
        # return self._input
        

    @property
    def output(self):
        r"""
        A mapping the output modes to their respective wires.
        """
        return self._output

    @property
    def adjoint(self) -> AdjointView:
        r"""The adjoint view of this Tensor (with new ``id``s). That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def modes(self) -> list[int]:
        r"""
        For backward compatibility. Don't overuse.
        It returns a list of modes for this Tensor, unless it's ambiguous.
        """
        if self.modes_in == self.modes_out:  # transformation on same modes
            return list(self.modes_in)
        elif len(self.modes_in) == 0:  # state
            return list(self.modes_out)
        elif len(self.modes_out) == 0:  # measurement
            return list(self.modes_in)
        else:
            raise ValueError("modes are ambiguous for this Tensor.")

    @modes.setter
    def modes(self, value: list[int]):
        r"""
        For backward compatibility. Don't overuse.
        It resets the modes and the wires for this Tensor.
        It cannot be used if input and output modes are not equal.
        """
        if self.modes_in == self.modes_out:  # transformation on same modes
            self._update_modes(
                modes_out_bra=value if self._modes_out_bra == self._modes_out_ket else None,
                modes_in_bra=value if self._modes_in_bra == self._modes_in_ket else None,
                modes_out_ket=value,
                modes_in_ket=value,
            )
        elif len(self.modes_in) == 0:  # state
            self._update_modes(
                modes_out_bra=value if self._modes_in_bra == self._modes_in_ket else None,
                modes_in_bra=[],
                modes_out_ket=value,
                modes_in_ket=[],
            )
        elif len(self.modes_out) == 0:  # measurement
            self._update_modes(
                modes_out_bra=[],
                modes_in_bra=value if self._modes_out_bra == self._modes_out_ket else None,
                modes_out_ket=[],
                modes_in_ket=value,
            )
        else:
            raise ValueError("modes are ambiguous for this Tensor.")

    @property
    def modes_in(self) -> List[int]:
        r"""
        The list of input modes that are used by this Tensor.

        If this tensor has no input modes on the bra side, or if the input modes are equal
        on both ket and bra sides, it returns the list of modes. Otherwise, it performs the
        ``set()`` operation before returning the list (and hence, the order may be unexpected).
        """
        return self._modes_in_ket if self._modes_in_ket else self._modes_in_bra

    @property
    def modes_out(self) -> List[int]:
        r"""
        The list of output modes that are used by this Tensor.

        If this tensor has no output modes on the bra side, or if the output modes are equal
        on both ket and bra sides, it returns the list of modes. Otherwise, it performs the
        ``set()`` operation before returning the list (and hence, the order may be unexpected).
        """
        return self._modes_out_ket if self._modes_out_ket else self._modes_out_bra

    def unpack_shape(self, shape: Tuple[int]):
        r"""
        Unpack the given ``shape`` into the shapes of the input and output wires on ket and bra sides.

        Args:
            shape: A shape.

        Returns:
            shape_in_ket: The shape of the input wires on the ket side.
            shape_out_ket: The shape of the output wires on the ket side.
            shape_in_bra: The shape of the input wires on the bra side.
            shape_out_bra: The shape of the output wires on the bra side.
        """
        idx1 = 0
        idx2 = len(self._modes_in_ket)
        shape_in_ket = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_out_ket)
        shape_out_ket = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_in_bra)
        shape_in_bra = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_out_bra)
        shape_out_bra = shape[idx1:idx2]

        return shape_in_ket, shape_out_ket, shape_in_bra, shape_out_bra

    @property
    def wires(self) -> List[Wire]:
        r"""
        The list of all wires in this tensor, sorted as ``[ket_in, ket_out, bra_in, bra_out]``.
        """
        return (
            list(self.input.ket.values())
            + list(self.output.ket.values())
            + list(self.input.bra.values())
            + list(self.output.bra.values())
        )

    @abstractmethod
    def value(self, shape: Tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of this tensor

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """

    def change_modes(
        self,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ) -> None:
        r"""
        Changes the modes in this tensor.

        Args:
            name: The name of this tensor.
            modes_in_ket: The input modes on the ket side.
            modes_out_ket: The output modes on the ket side.
            modes_in_bra: The input modes on the bra side.
            modes_out_bra: The output modes on the bra side.

        Raises:
            ValueError: if one or more wires in this tensor are already connected.
        """
        for wire in self.wires:
            if wire.is_connected:
                msg = (
                    "Cannot change nodes in a tensor when some of its wires are already connected."
                )
                raise ValueError(msg)
        self._update_modes(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    def shape(self, default_dim: Optional[int] = None, out_in=False):
        r"""
        Returns the shape of the underlying tensor, as inferred from the dimensions of the individual
        wires.

        If ``out_in`` is ``False``, the shape returned is in the order ``(in_ket, in_bra, out_ket, out_bra)``.
        Otherwise, it is in the order ``(out_ket, out_bra, in_ket, in_bra)``.

        Args:
            default_dim: The default dimension of wires with unspecified dimension.
            out_in: Whether to return output shapes followed by input shapes or viceversa.
        """

        def _sort_shapes(*args):
            for arg in args:
                if arg:
                    yield arg

        shape_in_ket = [w.dim if w.dim else default_dim for w in self.input.ket.values()]
        shape_out_ket = [w.dim if w.dim else default_dim for w in self.output.ket.values()]
        shape_in_bra = [w.dim if w.dim else default_dim for w in self.input.bra.values()]
        shape_out_bra = [w.dim if w.dim else default_dim for w in self.output.bra.values()]

        if out_in:
            ret = _sort_shapes(shape_out_ket, shape_out_bra, shape_in_ket, shape_in_bra)
        ret = _sort_shapes(shape_in_ket, shape_in_bra, shape_out_ket, shape_out_bra)

        # pylint: disable=consider-using-generator
        return tuple([item for sublist in ret for item in sublist])


class AdjointView(Tensor):
    r"""
    Adjoint view of a tensor. It swaps the ket and bra wires of a Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            name=self._original.name,
            modes_in_ket=self._original.input.bra.keys(),
            modes_out_ket=self._original.output.bra.keys(),
            modes_in_bra=self._original.input.ket.keys(),
            modes_out_bra=self._original.output.ket.keys(),
        )

    def value(self, shape: Tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of the adjoint tensor.

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """
        # converting the given shape into a shape for the original tensor
        shape_in_ket, shape_out_ket, shape_in_bra, shape_out_bra = self._original.unpack_shape(
            shape
        )
        shape_ret = shape_in_bra + shape_out_bra + shape_in_ket + shape_out_ket

        return math.conj(math.astensor(self._original.value(shape_ret)))


class DualView(Tensor):
    r"""
    Dual view of a tensor. It swaps the input and output wires of a Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            name=self._original.name,
            modes_in_ket=self._original.output.ket.keys(),
            modes_out_ket=self._original.input.ket.keys(),
            modes_in_bra=self._original.output.bra.keys(),
            modes_out_bra=self._original.input.bra.keys(),
        )

    def value(self, shape: Tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of the dual tensor.

        Returns:
            ComplexTensor: the unitary matrix in Fock representation.
        """
        # converting the given shape into a shape for the original tensor
        shape_in_ket, shape_out_ket, shape_in_bra, shape_out_bra = self.unpack_shape(shape)
        shape_ret = shape_out_ket + shape_in_ket + shape_out_bra, shape_in_bra

        return math.conj(self._original.value(shape_ret))
