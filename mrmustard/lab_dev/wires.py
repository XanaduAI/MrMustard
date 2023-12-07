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

from typing import Optional, Sequence, Union

import uuid

__all__ = [
    "Wires",
]


def random_int() -> int:
    r"""
    Returns a random integer obtained from a UUID
    """
    return uuid.uuid1().int


class WiresLog:
    r"""
    A wrap around a dictionary that maps modes to a wire ``id`` or ``None``.

    Arguments:
        modes: All the modes in this ``WireLog``.
        modes_with_id: The modes that need to be given an ``id``.
    """

    def __init__(
        self, modes: Optional[Sequence[int]] = None, modes_with_id: Optional[Sequence[int]] = None
    ) -> None:
        modes = modes or []
        modes_with_id = modes_with_id or []
        if not set(modes_with_id).issubset(modes):
            msg = f"{modes_with_id} is not subset of {modes}"
            raise ValueError(msg)
        self._log = {m: uuid.uuid1().int if m in modes_with_id else None for m in modes}

    @property
    def log(self) -> dict[int, Optional[int]]:
        r"""
        The dictionary mapping modes to ``id``s wrapped in this ``WireLog``.
        """
        return self._log

    def remove_ids(self) -> WiresLog:
        r"""
        Returns a new ``WiresLog`` where all the ``id``s in the log are replaced with ``None``.
        """
        return WiresLog(list(self._log.keys()))

    def __getitem__(self, modes: Sequence[int]) -> WiresLog:
        r"""
        Returns a new ``WiresLog`` with wires on remaining modes set to ``None``.
        """
        ret = WiresLog()
        ret._log = {m: (v if m in modes else None) for m, v in self._log.items()}
        return ret


class Wires:
    r"""
    A class with wire functionality for tensor network tensor applications.

    Each wire is characterized by a unique identifier ``id``, which must be different from
    the identifiers of all the other wires in the tensor network. Additionally, it owns a
    label ``mode`` that represents the mode of light that this wire is acting on.

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
        msg = "Modes on ket and bra sides must be equal, unless either of them is ``None``."
        if modes_in_ket and modes_in_bra:
            if modes_in_ket != modes_in_bra:
                msg = f"Input {msg}"
                raise ValueError(msg)
        if modes_out_ket and modes_out_bra:
            if modes_out_ket != modes_out_bra:
                msg = f"Output {msg}"
                raise ValueError(msg)

        all_modes = (
            (modes_in_ket or [])
            + (modes_out_ket or [])
            + (modes_in_bra or [])
            + (modes_out_bra or [])
        )
        self._modes = list[set(all_modes)]

        self._in_ket = WiresLog(self._modes, modes_in_ket)
        self._out_ket = WiresLog(self._modes, modes_out_ket)
        self._in_bra = WiresLog(self._modes, modes_in_bra)
        self._out_bra = WiresLog(self._modes, modes_out_bra)

    @classmethod
    def from_wires(
        cls, in_ket: WiresLog, out_ket: WiresLog, in_bra: WiresLog, out_bra: WiresLog
    ) -> Wires:
        r"""
        Initializes a ``Wires`` object from ``WireLog`` objects.
        """
        ret = Wires()
        ret._in_ket = in_ket
        ret._out_ket = out_ket
        ret._in_bra = in_bra
        ret._out_bra = out_bra
        return ret

    @property
    def input(self) -> Wires:
        r"""
        Returns a new tensor where the output wires have no ``id``s.
        """
        return self.from_wires(
            self._in_ket,
            self._out_ket.remove_ids(),
            self._in_bra,
            self._out_bra.remove_ids(),
        )

    @property
    def output(self) -> Wires:
        r"""
        Returns a new tensor where the input wires have no ``id``s.
        """
        return self.from_wires(
            self._in_ket.remove_ids(),
            self._out_ket,
            self._in_bra.remove_ids(),
            self._out_bra,
        )

    @property
    def ket(self) -> Wires:
        r"""
        Returns a new tensor where the bra wires have no ``id``s.
        """
        return self.from_wires(
            self._in_ket,
            self._out_ket,
            self._in_bra.remove_ids(),
            self._out_bra.remove_ids(),
        )

    @property
    def bra(self) -> Wires:
        r"""
        Returns a new tensor where the ket wires have no ``id``s.
        """
        return self.from_wires(
            self._in_ket.remove_ids(),
            self._out_ket.remove_ids(),
            self._in_bra,
            self._out_bra,
        )

    def __getitem__(self, modes: Union[int, Sequence[int]]) -> Wires:
        r"""
        Returns a new ``Wires`` object with wires on remaining modes set to ``None``.
        """
        modes = [modes] if isinstance(modes, int) else modes
        return self.from_wires(
            self._in_ket[modes],
            self._out_ket[modes],
            self._in_bra[modes],
            self._out_bra[modes],
        )

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
        The modes in this ``Wires``.
        """
        return self._modes

    @property
    def modes_out(self) -> list[int]:
        r"""
        Returns a list of output modes for this ``Wires``.
        """
        return list({m: None for m in self._out_bra | self._out_ket}.keys())

    @property
    def modes_in(self) -> list[int]:
        r"""
        Returns a list of input modes on the bra side for this ``Wires``.
        """
        return list({m: None for m in self._in_bra | self._in_ket}.keys())

    @property
    def modes_ket(self) -> list[int]:
        r"""
        Returns a list of output modes on the ket side for this ``Wires``.
        """
        return list({m: None for m in self._out_ket | self._in_ket}.keys())

    # @property
    # def ids(self) -> list[int | None]:
    #     r"""
    #     Returns a list of ids for this Tensor for id position search.
    #     """
    #     return (
    #         list(self._out_bra.values())
    #         + list(self._in_bra.values())
    #         + list(self._out_ket.values())
    #         + list(self._in_ket.values())
    #     )

    # def __bool__(self):
    #     r"""Make self truthy if any of the ids are not None and falsy otherwise"""
    #     return any(self.ids)
