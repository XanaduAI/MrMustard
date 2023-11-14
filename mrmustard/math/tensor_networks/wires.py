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

import uuid

class Wires:
    r"""A class with wire functionality for tensor network tensor applications.
    Anything that wants wires should use an object of this class.

    In general we distinguish between input and output wires, and between ket and bra sides.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: list[int] = [],
        modes_in_bra: list[int] = [],
        modes_out_ket: list[int] = [],
        modes_in_ket: list[int] = [],
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

        union = {
            m: None for m in modes_out_bra + modes_in_bra + modes_out_ket + modes_in_ket
        }.keys()
        self._out_bra = {
            m: uuid.uuid1().int if m in modes_out_bra else None for m in union
        }
        self._in_bra = {
            m: uuid.uuid1().int if m in modes_in_bra else None for m in union
        }
        self._out_ket = {
            m: uuid.uuid1().int if m in modes_out_ket else None for m in union
        }
        self._in_ket = {
            m: uuid.uuid1().int if m in modes_in_ket else None for m in union
        }

    @classmethod
    def from_wires(cls, out_bra, in_bra, out_ket, in_ket) -> Wires:
        new = Wires()
        new._out_bra = out_bra
        new._out_ket = out_ket
        new._in_bra = in_bra
        new._in_ket = in_ket
        return new

    @property
    def input(self) -> Wires:
        r"""
        Returns a new tensor with output wires set to None
        """
        return self.from_wires(
            self._out_bra,
            {m: None for m in self._in_bra},
            self._out_ket,
            {m: None for m in self._in_ket},
        )

    @property
    def output(self) -> Wires:
        r"""
        Returns a new tensor with input wires set to None
        """
        return self.from_wires(
            {m: None for m in self._out_bra},
            self._in_bra,
            {m: None for m in self._out_ket},
            self._in_ket,
        )

    @property
    def ket(self) -> Wires:
        r"""
        Returns a new tensor with bra wires set to None
        """
        return self.from_wires(
            self._out_bra,
            self._in_bra,
            {m: None for m in self._out_ket},
            {m: None for m in self._in_ket},
        )

    @property
    def bra(self) -> Wires:
        r"""
        Returns a new tensor with ket wires set to None
        """
        return self.from_wires(
            {m: None for m in self._out_bra},
            {m: None for m in self._in_bra},
            self._out_ket,
            self._in_ket,
        )

    def __getitem__(self, modes) -> Wires:
        r"""
        Returns a new tensor with wires on remaining modes set to None
        """
        modes = [modes] if isinstance(modes, int) else modes
        return self.from_wires(
            {m: None for m in self._out_bra if m not in modes},
            {m: None for m in self._in_bra if m not in modes},
            {m: None for m in self._out_ket if m not in modes},
            {m: None for m in self._in_ket if m not in modes},
        )

    @property
    def adjoint(self) -> Wires:
        r"""The adjoint view of this Tensor (with new ``id``s). That is, ket <-> bra."""
        return self.from_wires(self._out_ket, self._in_ket, self._out_bra, self._in_bra)

    @property
    def dual(self) -> Wires:
        r"""The dual view of this Tensor (with new ``id``s). That is, input <-> output."""
        return self.from_wires(self._in_bra, self._out_bra, self._in_ket, self._out_ket)

    @property
    def modes(self) -> list[int]:
        r"""
        For backward compatibility. Don't overuse.
        It returns a list of modes for this Tensor.
        """
        return list(
            {
                m: None
                for m in self._out_bra | self._in_bra | self._out_ket | self._in_ket
            }.keys()
        )

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
    
    def __bool__(self):
        r"""Evaluate self to True if any of the ids are not None and False otherwise"""
        return any(self.ids)
