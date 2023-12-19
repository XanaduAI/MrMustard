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
import uuid

from ..utils.typing import Mode


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
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.

    .. jupyter-execute::

        from mrmustard.lab_dev.wires import Wires

        # initialize modes
        modes = [0, 1]

        # initialize `wires` object with given modes
        wires = Wires(modes_out_ket = modes, modes_in_ket = modes)
    """

    def __init__(
        self,
        modes_out_bra: Optional[Sequence[Mode]] = None,
        modes_in_bra: Optional[Sequence[Mode]] = None,
        modes_out_ket: Optional[Sequence[Mode]] = None,
        modes_in_ket: Optional[Sequence[Mode]] = None,
    ) -> None:
        modes_out_bra = modes_out_bra or []
        modes_in_bra = modes_in_bra or []
        modes_out_ket = modes_out_ket or []
        modes_in_ket = modes_in_ket or []

        modes = modes_out_bra or modes_in_bra or modes_out_ket or modes_in_ket
        if (
            (modes_out_bra and modes_out_bra != modes)
            or (modes_in_bra and modes_in_bra != modes)
            or (modes_out_ket and modes_out_ket != modes)
            or (modes_in_ket and modes_in_ket != modes)
        ):
            # cannot define the list of modes unambiguously
            self._modes = None
        else:
            self._modes = list(modes)

        keys = self._modes or set(modes_in_ket + modes_out_ket + modes_in_bra + modes_out_bra)
        self._out_bra = {m: uuid.uuid4().int if m in modes_out_bra else None for m in keys}
        self._in_bra = {m: uuid.uuid4().int if m in modes_in_bra else None for m in keys}
        self._out_ket = {m: uuid.uuid4().int if m in modes_out_ket else None for m in keys}
        self._in_ket = {m: uuid.uuid4().int if m in modes_in_ket else None for m in keys}

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
    def modes(self) -> list[Mode]:
        r"""
        The list of all the ``Mode``s in this ``Wires``.
        """
        if not self._modes:
            msg = "Cannot return the list of modes unambiguously."
            raise ValueError(msg)
        return self._modes

    @property
    def modes_out_bra(self) -> list[Mode]:
        r"""
        The list of all the output ``Mode``s in this ``Wires`` on the bra side.
        """
        return [m for m, w in self.out_bra.items() if w is not None]

    @property
    def modes_in_bra(self) -> list[Mode]:
        r"""
        The list of all the input ``Mode``s in this ``Wires`` on the bra side.
        """
        return [m for m, w in self.in_bra.items() if w is not None]

    @property
    def modes_out_ket(self) -> list[Mode]:
        r"""
        The list of all the output ``Mode``s in this ``Wires`` on the ket side.
        """
        return [m for m, w in self.out_ket.items() if w is not None]

    @property
    def modes_in_ket(self) -> list[Mode]:
        r"""
        The list of all the input ``Mode``s in this ``Wires`` on the ket side.
        """
        return [m for m, w in self.in_ket.items() if w is not None]

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
        modes_out_bra = [m for m, w in self.out_bra.items() if w]
        modes_in_bra = [m for m, w in self.in_bra.items() if w]
        modes_out_ket = [m for m, w in self.out_ket.items() if w]
        modes_in_ket = [m for m, w in self.in_ket.items() if w]
        return Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)

    def __getitem__(self, modes: Union[Mode, Sequence[Mode]]) -> Wires:
        r"""
        Returns a copy of this ``Wires`` with only the given modes. It does not
        change the ``id``s.
        """
        modes = [modes] if isinstance(modes, Mode) else modes
        ret = Wires()
        ret._out_bra = {m: k for m, k in self._out_bra.items() if m in modes}
        ret._in_bra = {m: k for m, k in self._in_bra.items() if m in modes}
        ret._out_ket = {m: k for m, k in self._out_ket.items() if m in modes}
        ret._in_ket = {m: k for m, k in self._in_ket.items() if m in modes}
        return ret
