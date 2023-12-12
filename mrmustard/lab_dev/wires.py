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

        self._out_ket = {m: uuid.uuid4() if m in modes_out_ket else None for m in self._modes}
        self._in_bra = {m: uuid.uuid4() if m in modes_in_bra else None for m in self._modes}
        self._out_bra = {m: uuid.uuid4() if m in modes_out_bra else None for m in self._modes}
        self._in_ket = {m: uuid.uuid4() if m in modes_in_ket else None for m in self._modes}

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
        modes_in_ket = [m for m, w in self.in_ket.items() if w]
        modes_out_ket = [m for m, w in self.out_ket.items() if w]
        modes_in_bra = [m for m, w in self.in_bra.items() if w]
        modes_out_bra = [m for m, w in self.out_bra.items() if w]
        return Wires(modes_in_bra, modes_out_bra, modes_in_ket, modes_out_ket)

    @property
    def modes(self) -> set[Mode]:
        r"""
        The set of all the ``Mode``s (input, output, ket, and bra) in this ``Wires``.
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
