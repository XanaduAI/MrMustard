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

"""
Classes to represent wires for tensor network applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import uuid

__all__ = [
    "Wires",
]


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
    A list of ``Wire`` objects.

    Arguments:
        modes: All the modes in this ``Wires``.
    """

    def __init__(self, modes: Optional[Sequence[int]] = None) -> None:
        modes = modes or []
        self._items = {m: Wire(m) for m in modes}

    @property
    def modes(self):
        r"""
        The modes in this ``Wires``.
        """
        return list(self._items.keys())

    @property
    def items(self):
        r"""
        The ``Wire``s in this ``Wires``.
        """
        return list(self._items.values())

    def __getitem__(self, idx: int):
        return self._items[idx]


class Wired:
    r"""
    A class representing a wired object for tensor network tensor applications.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
    """

    def __init__(
        self,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ) -> None:
        msg = "modes on ket and bra sides must be equal, unless either of them is ``None``."
        if modes_in_ket and modes_in_bra:
            if modes_in_ket != modes_in_bra:
                msg = f"Input {msg}"
                raise ValueError(msg)
        if modes_out_ket and modes_out_bra:
            if modes_out_ket != modes_out_bra:
                msg = f"Output {msg}"
                raise ValueError(msg)

        self._modes = list(
            set(
                (modes_in_ket or [])
                + (modes_out_ket or [])
                + (modes_in_bra or [])
                + (modes_out_bra or [])
            )
        )

        self._in_ket = Wires(modes_in_ket)
        self._out_ket = Wires(modes_out_ket)
        self._in_bra = Wires(modes_in_bra)
        self._out_bra = Wires(modes_out_bra)

    @classmethod
    def from_wires(cls, in_ket: Wires, out_ket: Wires, in_bra: Wires, out_bra: Wires) -> Wires:
        r"""
        Initializes a ``Wires`` object from ``WireLog`` objects.
        """
        ret = Wired()
        ret._in_ket = in_ket
        ret._in_ket = out_ket
        ret._in_bra = in_bra
        ret._out_bra = out_bra
        return ret

    def new(self) -> Wired:
        r"""
        Returns a new instance of this ``Wired`` with new ``id``s.
        """
        modes_in_ket = self.in_ket.modes
        modes_out_ket = self.out_ket.modes
        modes_in_bra = self.in_bra.modes
        modes_out_bra = self.out_bra.modes
        return Wired(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    @property
    def in_ket(self) -> Wires:
        r"""
        The ``Wires`` for the input modes on the ket side.
        """
        return self._in_ket

    @property
    def out_ket(self) -> Wires:
        r"""
        The ``Wires`` for the output modes on the ket side.
        """
        return self._out_ket

    @property
    def in_bra(self) -> Wires:
        r"""
        The ``Wires`` for the input modes on the bra side.
        """
        return self._in_bra

    @property
    def out_bra(self) -> Wires:
        r"""
        The ``Wires`` for the output modes on the bra side.
        """
        return self._out_bra

    @property
    def adjoint(self) -> Wires:
        r"""The adjoint view of this Wires. That is, ket <-> bra."""
        return self.from_wires(
            self._in_bra,
            self._out_bra,
            self._in_ket,
            self._out_ket,
        )

    @property
    def dual(self) -> Wires:
        r"""The dual view of this ``Wires``. That is, input <-> output."""
        return self.from_wires(
            self._out_ket,
            self._in_ket,
            self._out_bra,
            self._in_bra,
        )

    @property
    def modes(self) -> list[int]:
        r"""
        A sorted list of all the modes in this ``Wires``.
        """
        return self._modes
