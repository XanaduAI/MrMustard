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

""" Classes for supporting tensor network functionalities."""

from __future__ import annotations

from typing import Optional, Sequence, Union

from ..utils.typing import Mode

import uuid

__all__ = ["Wire", "Wires"]

Wire = int
r"""
An integer representing a wire in a tensor network.
"""


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
        modes_in_ket = modes_in_ket or []
        modes_out_ket = modes_out_ket or []
        modes_in_bra = modes_in_bra or []
        modes_out_bra = modes_out_bra or []

        modes = modes_in_ket or modes_out_ket or modes_in_bra or modes_out_bra
        if (
            (modes_in_ket and modes_in_ket != modes)
            or (modes_out_ket and modes_out_ket != modes)
            or (modes_in_bra and modes_in_bra != modes)
            or (modes_out_bra and modes_out_bra != modes)
        ):
            # cannot define the list of modes unambiguously
            self._modes = None
        else:
            self._modes = modes
        self._set_modes = set(modes) or set(
            modes_in_ket + modes_out_ket + modes_in_bra + modes_out_bra
        )

        keys = self._modes or self._set_modes
        self._out_ket = {m: uuid.uuid4() if m in modes_out_ket else None for m in keys}
        self._in_bra = {m: uuid.uuid4() if m in modes_in_bra else None for m in keys}
        self._out_bra = {m: uuid.uuid4() if m in modes_out_bra else None for m in keys}
        self._in_ket = {m: uuid.uuid4() if m in modes_in_ket else None for m in keys}

        self.modes_in_ket = modes_in_ket
        self.modes_out_ket = modes_out_ket
        self.modes_in_bra = modes_in_bra
        self.modes_out_bra = modes_out_bra

    def list_of_types_and_modes_of_wires(self):
        r'''gives the list of types and modes for each wires in bargmann representation.'''
        list_types = []
        list_modes = []
        for m in self.modes:
            if m in self.modes_out_bra:
                list_types.append('out_bra')
                list_modes.append(m)
            if m in self.modes_in_bra:
                list_types.append('in_bra')
                list_modes.append(m)
            if m in self.modes_out_ket:
                list_types.append('out_ket')
                list_modes.append(m)
            if m in self.modes_in_ket:
                list_types.append('in_ket')
                list_modes.append(m)
        return list_types, list_modes

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
    def modes(self) -> list[Mode]:
        r"""
        The list of all the ``Mode``s in this ``Wires``.
        """
        if not self._modes:
            msg = "Cannot return the list of modes unambiguously."
            raise ValueError(msg)
        return self._modes

    @property
    def has_bra(self) -> bool:
        r"""
        Whether this ``Wires`` has wires on the bra side.
        """
        has_in = len([m for m, w in self.in_bra.items() if w]) > 0
        has_out = len([m for m, w in self.out_bra.items() if w]) > 0
        return has_in or has_out

    @property
    def set_modes(self) -> set[Mode]:
        r"""
        A set of all the ``Mode``s in this ``Wires``.
        """
        return self._set_modes

    def adjoint(self) -> Wires:
        r"""
        The adjoint of this ``Wires`` (with new ``id``s), obtained switching kets and bras.
        """
        modes_in_ket = [m for m, w in self.in_ket.items() if w]
        modes_out_ket = [m for m, w in self.out_ket.items() if w]
        modes_in_bra = [m for m, w in self.in_bra.items() if w]
        modes_out_bra = [m for m, w in self.out_bra.items() if w]
        return Wires(modes_in_bra, modes_out_bra, modes_in_ket, modes_out_ket)

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
        Returns a copy of this ``Wires`` with only the given modes. It does not
        change the ``id``s.
        """
        modes = [modes] if isinstance(modes, Mode) else modes
        ret = Wires()
        ret._in_ket = {m: k for m, k in self._in_ket.items() if m in modes}
        ret._out_ket = {m: k for m, k in self._out_ket.items() if m in modes}
        ret._in_bra = {m: k for m, k in self._in_bra.items() if m in modes}
        ret._out_bra = {m: k for m, k in self._out_bra.items() if m in modes}
        return ret
