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

from typing import Optional, Sequence, Union

import uuid

class Wire:
    r"""
    A class to represent individual wires in a tensor network.

    Each wire owns a label ``mode`` that represents the mode of light that the wire
    is acting on. Moreover, it is characterized by a numerical identifier ``id``,
    as well as by a boolean ``is_connected`` that is ``True`` if this wire has been
    connected to another wire, and ``False`` otherwise.

    Arguments:
        mode: The mode of light the wire is acting on.
    """

    def __init__(self, mode: int) -> None:
        self._mode = mode
        self._id = uuid.uuid1().int
        self._is_connected = False

    @property
    def id(self) -> int:
        r"""
        The numerical identifier of this wire.
        """
        return self._id

    @id.setter
    def id(self, value: int):
        if self.is_connected:
            msg = "Cannot change the ``id`` or a wire that is already connected."
            raise ValueError(msg)
        self._id = value

    @property
    def is_connected(self) -> bool:
        r"""
        Whether or not this wire is connected with another wire.
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value

    @property
    def mode(self) -> int:
        r"""
        The mode represented by this wire.
        """
        return self._mode

    def connect(self, other: Wire) -> None:
        r"""
        Connects this wire with another wire.

        Raises an error if one or both wires are already connected with a
        different wires, and does nothing if the two wires are already connected
        together.

        Arguments:
            other: Another wire.

        Raises:
            ValueError: If one or both wires are already connected with a
                different wires.
        """
        if self.id != other.id:
            if not (self.is_connected or other.is_connected):
                other.id = self.id
                self.is_connected = True
                other.is_connected = True
            else:
                msg = "Cannot connect a wire that is already connected."
                raise ValueError(msg)

class Wires:
    r"""
    A class with wire functionality for tensor network tensor applications.

    In general we distinguish between input and output wires, and between ket and bra sides.

    Args:
        modes_in_ket: The input modes on the ket side.
        modes_out_ket: The output modes on the ket side.
        modes_in_bra: The input modes on the bra side.
        modes_out_bra: The output modes on the bra side.
    """

    def __init__(
        self,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ) -> None:
        msg = (
            "modes on ket and bra sides must be equal, unless either of them is `None`."
        )
        if modes_in_ket and modes_in_bra:
            if modes_in_ket != modes_in_bra:
                msg = f"Input {msg}"
                raise ValueError(msg)
        if modes_out_ket and modes_out_bra:
            if modes_out_ket != modes_out_bra:
                msg = f"Output {msg}"
                raise ValueError(msg)
            
        modes_in_ket = modes_in_ket or []
        modes_out_ket = modes_out_ket or []
        modes_in_bra = modes_in_bra or []
        modes_out_bra = modes_out_bra or []
        self._modes = set(modes_in_ket + modes_out_ket + modes_in_bra + modes_out_bra)

        self._out_ket = {
            m: Wire(m) if m in modes_out_ket else None for m in self._modes
        }
        self._in_bra = {
            m: Wire(m) if m in modes_in_bra else None for m in self._modes
        }
        self._out_bra = {
            m: Wire(m) if m in modes_out_bra else None for m in self._modes
        }
        self._in_ket = {
            m: Wire(m) if m in modes_in_ket else None for m in self._modes
        }

    @classmethod
    def from_wires(cls, in_ket, out_ket, in_bra, out_bra) -> Wires:
        new = Wires()
        new._in_ket = in_ket
        new._in_bra = in_bra
        new._out_ket = out_ket
        new._out_bra = out_bra
        return new

    @property
    def in_ket(self) -> dict[int, Optional[Wire]]:
        r"""
        A dictionary mapping an integer ``m`` to a ``Wire`` if mode ``m`` has an
        input wire on the ket side, and to ``None`` otherwise.
        """
        return self._in_ket

    @property
    def out_ket(self) -> dict[int, Optional[Wire]]:
        r"""
        A dictionary mapping an integer ``m`` to a ``Wire`` if mode ``m`` has an
        ouput wire on the ket side, and to ``None`` otherwise.
        """
        return self._out_ket

    @property
    def in_bra(self) -> dict[int, Optional[Wire]]:
        r"""
        A dictionary mapping an integer ``m`` to a ``Wire`` if mode ``m`` has an
        input wire on the bra side, and to ``None`` otherwise.
        """
        return self._in_bra

    @property
    def out_bra(self) -> Wires:
        r"""
        A dictionary mapping an integer ``m`` to a ``Wire`` if mode ``m`` has an
        ouput wire on the bra side, and to ``None`` otherwise.
        """
        return self._out_bra

    @property
    def input(self) -> Wires:
        r"""
        Returns a new tensor with output wires set to None
        """
        return self.from_wires(
            {m: None for m in self._in_ket},
            self._out_ket,
            {m: None for m in self._in_bra},
            self._out_bra,
        )

    @property
    def output(self) -> Wires:
        r"""
        Returns a new tensor with input wires set to None
        """
        return self.from_wires(
            self._in_ket,
            {m: None for m in self._out_ket},
            self._in_bra,
            {m: None for m in self._out_bra},
        )

    @property
    def ket(self) -> Wires:
        r"""
        Returns a new tensor with bra wires set to None
        """
        return self.from_wires(
            {m: None for m in self._in_ket},
            {m: None for m in self._out_ket},
            self._in_bra,
            self._out_bra,
        )

    @property
    def bra(self) -> Wires:
        r"""
        Returns a new tensor with ket wires set to None
        """
        return self.from_wires(
            self._in_ket,
            self._out_ket,
            {m: None for m in self._in_bra},
            {m: None for m in self._out_bra},
        )

    @property
    def adjoint(self) -> Wires:
        r"""The adjoint of this ``Wires`` (with new ``id``s). That is, ket <-> bra."""
        return self.from_wires(self._in_bra, self._out_bra, self._in_ket, self._out_ket)

    @property
    def modes(self) -> set[int]:
        r"""
        The set of all the modes (input, output, ket, and bra) in this ``Wires``.
        """
        return self._modes

    @property
    def modes_out(self) -> list[int]:
        r"""
        It returns a list of output modes for this Tensor.
        """
        return list({m: None for m in self._out_bra | self._out_ket}.keys())
    
    @property
    def modes_in(self) -> list[int]:
        r"""
        It returns a list of input modes on the bra side for this Tensor.
        """
        return list({m: None for m in self._in_bra | self._in_ket}.keys())
    
    @property
    def modes_ket(self) -> list[int]:
        r"""
        It returns a list of output modes on the ket side for this Tensor.
        """
        return list({m: None for m in self._out_ket | self._in_ket}.keys())
    
    @property
    def ids(self) -> list[int | None]:
        r"""
        It returns a list of ids for this Tensor for id position search.
        """
        return (
            list(self._out_bra.values())
            + list(self._in_bra.values())
            + list(self._out_ket.values())
            + list(self._in_ket.values())
        )
    
    def new(self) -> Wires:
        r"""
        Returns a new instance of this ``Wires`` with new ``id``s.
        """
        modes_in_ket = [m for m, w in self.in_ket.items() if w]
        modes_out_ket = [m for m, w in self.out_ket.items() if w]
        modes_in_bra = [m for m, w in self.in_bra.items() if w]
        modes_out_bra = [m for m, w in self.out_bra.items() if w]
        return Wires(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
    
    def __bool__(self):
        r"""Make self truthy if any of the ids are not None and falsy otherwise"""
        return any(self.ids)

    def __getitem__(self, modes) -> Wires:
        r"""
        Returns a new ``Wires`` with wires on remaining modes set to ``None``.
        """
        modes = [modes] if isinstance(modes, int) else modes
        return self.from_wires(
            {m: k for m, k in self._in_ket.items() if m in modes},
            {m: k for m, k in self._out_ket.items() if m in modes},
            {m: k for m, k in self._in_bra.items() if m in modes},
            {m: k for m, k in self._out_bra.items() if m in modes},
        )