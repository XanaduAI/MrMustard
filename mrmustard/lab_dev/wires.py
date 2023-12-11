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
from mrmustard.utils.typing import Mode

import uuid


class Wire:
    r"""
    A class to represent individual wires in a tensor network.

    Each wire owns a label ``mode`` that represents the mode of light that the wire
    is acting on. Moreover, it is characterized by a numerical identifier ``id``.
    Initially, this ``id`` is randomly generated to be different from that of all the
    other wires initialized in the same session. However, when two wires are connected,
    their ``id``s are set equal.

    Arguments:
        mode: The mode of light the wire is acting on.
    """

    def __init__(self, mode: Mode) -> None:
        self._mode = mode
        self._id = uuid.uuid1().int
        self._is_connected = False
        # self._is_ket = ..
        # self._is_input = ..

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
    def mode(self) -> Mode:
        r"""
        The mode represented by this wire.
        """
        return self._mode

    def connect(self, other: Wire) -> None:
        r"""
        Connects this wire with another wire, or does nothing if the two wires are
        already connected with each other.

        Arguments:
            other: Another wire.

        Raises:
            ValueError: If one or both wires are already connected with different
                wires.
        """
        # think about adding more error checking to make sure we can only connect
        # correctly. E.g.:
        # - Can we connect wire on mode 1 with wire on mode 3?
        # - Can we connect two input wires?
        # - Can we connect a ket with a bra?
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
        modes_in_ket: Optional[Sequence[Mode]] = None,
        modes_out_ket: Optional[Sequence[Mode]] = None,
        modes_in_bra: Optional[Sequence[Mode]] = None,
        modes_out_bra: Optional[Sequence[Mode]] = None,
    ) -> None:
        msg = "modes on ket and bra sides must be equal, unless either of them is `None`."
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

        self._out_ket = {m: Wire(m) if m in modes_out_ket else None for m in self._modes}
        self._in_bra = {m: Wire(m) if m in modes_in_bra else None for m in self._modes}
        self._out_bra = {m: Wire(m) if m in modes_out_bra else None for m in self._modes}
        self._in_ket = {m: Wire(m) if m in modes_in_ket else None for m in self._modes}

    @classmethod
    def from_wires(
        cls,
        in_ket: dict[Mode, Wire],
        out_ket: dict[Mode, Wire],
        in_bra: dict[Mode, Wire],
        out_bra: dict[Mode, Wire],
    ) -> Wires:
        r"""
        Initializes a new ``Wires`` object from dictionaries.

        Arguments:
            ``in_ket``: A dictionary mapping input modes on the ket side to wires.
            ``out_ket``: A dictionary mapping output modes on the ket side to wires.
            ``in_bra``: A dictionary mapping input modes on the bra side to wires.
            ``out_ket``: A dictionary mapping output modes on the bra side to wires.
        """
        # Consider removing, as this can be buggy
        ret = Wires()
        ret._in_ket = in_ket
        ret._in_bra = in_bra
        ret._out_ket = out_ket
        ret._out_bra = out_bra
        return ret

    @property
    def in_ket(self) -> dict[Mode, Optional[Wire]]:
        r"""
        A dictionary mapping a mode ``m`` to a ``Wire`` if mode ``m`` has an
        input wire on the ket side, and to ``None`` otherwise.
        """
        return self._in_ket

    @property
    def out_ket(self) -> dict[Mode, Optional[Wire]]:
        r"""
        A dictionary mapping a mode ``m`` to a ``Wire`` if mode ``m`` has an
        ouput wire on the ket side, and to ``None`` otherwise.
        """
        return self._out_ket

    @property
    def in_bra(self) -> dict[Mode, Optional[Wire]]:
        r"""
        A dictionary mapping a mode ``m`` to a ``Wire`` if mode ``m`` has an
        input wire on the bra side, and to ``None`` otherwise.
        """
        return self._in_bra

    @property
    def out_bra(self) -> dict[Mode, Optional[Wire]]:
        r"""
        A dictionary mapping a mode ``m`` to a ``Wire`` if mode ``m`` has an
        ouput wire on the bra side, and to ``None`` otherwise.
        """
        return self._out_bra

    @property
    def adjoint(self) -> Wires:
        r"""
        The adjoint of this ``Wires`` (with new ``id``s), obtained switching kets and bras.
        """
        return self.from_wires(self._in_bra, self._out_bra, self._in_ket, self._out_ket)

    @property
    def modes(self) -> set[int]:
        r"""
        The set of all the modes (input, output, ket, and bra) in this ``Wires``.
        """
        return self._modes

    def new(self) -> Wires:
        r"""
        Returns a copy of this ``Wires`` with new ``id``s.
        """
        modes_in_ket = [m for m, w in self.in_ket.items() if w]
        modes_out_ket = [m for m, w in self.out_ket.items() if w]
        modes_in_bra = [m for m, w in self.in_bra.items() if w]
        modes_out_bra = [m for m, w in self.out_bra.items() if w]
        return Wires(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    def __getitem__(self, modes: Union[Mode, Sequence[Mode]]) -> Wires:
        r"""
        Returns a copy of this ``Wires`` with only the given modes.
        """
        modes = [modes] if isinstance(modes, Mode) else modes
        return self.from_wires(
            {m: k for m, k in self._in_ket.items() if m in modes},
            {m: k for m, k in self._out_ket.items() if m in modes},
            {m: k for m, k in self._in_bra.items() if m in modes},
            {m: k for m, k in self._out_bra.items() if m in modes},
        )
