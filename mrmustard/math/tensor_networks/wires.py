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
from typing import Iterable
import uuid

class Wires:
    r"""A class with wire functionality for tensor network applications.
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

        union = set(modes_out_bra).union(modes_in_bra).union(modes_out_ket).union(modes_in_ket)
        _out_bra = {m: uuid.uuid1().int if m in modes_out_bra else None for m in union}
        _in_bra = {m: uuid.uuid1().int if m in modes_in_bra else None for m in union}
        _out_ket = {m: uuid.uuid1().int if m in modes_out_ket else None for m in union}
        _in_ket = {m: uuid.uuid1().int if m in modes_in_ket else None for m in union}
        self._ids = {m: [_out_bra[m], _in_bra[m], _out_ket[m], _in_ket[m]] for m in union}
    
    def copy(self):
        "A copy that preserves the ids"
        w = Wires()
        w.__dict__ = self.__dict__.copy()
        return w

    @property
    def input(self) -> Wires:
        r"""
        A new Wires object without output wires
        """
        w = self.copy()
        w._ids = {m: [None, w._ids[m][1], None, w._ids[m][3]] for m in w._ids if any(w._ids[m])}
        return w

    @property
    def output(self) -> Wires:
        r"""
        A new Wires object without input wires
        """
        w = self.copy()
        w._ids = {m: [w._ids[m][0], None, w._ids[m][2], None] for m in w._ids if any(w._ids[m])}
        return w

    @property
    def ket(self) -> Wires:
        r"""
        A new Wires object without bra wires
        """
        w = self.copy()
        w._ids = {m: [None, None, w._ids[m][2], w._ids[m][3]] for m in w._ids if any(w._ids[m])}
        return w

    @property
    def bra(self) -> Wires:
        r"""
        A new Wires object without ket wires
        """
        w = self.copy()
        w._ids = {m: [w._ids[m][0], w._ids[m][1],None, None] for m in w._ids if any(w._ids[m])}
        return w

    def __getitem__(self, modes: Iterable[int] | int) -> Wires:
        r"""
        A new Wires object with wires on the given modes.
        """
        modes = [modes] if isinstance(modes, int) else modes
        w = self.copy()
        w._ids = {m: w._ids[m] for m in w._ids if m in modes}
        return w

    @property
    def modes(self) -> set[int]:
        r"""
        For backward compatibility. It returns the modes occupied by wires.
        """
        return set([m for m in self._ids if any(self._ids[m])])
    
    @property
    def ids(self) -> list[int | None]:
        r"""
        A list of ids for this Wires object. The order of the ids is
        bra.out + bra.in + ket.out + ket.in where each of the four groups is ordered by mode.
        """
        return [id for ids in zip(*self._ids.values()) for id in ids if id is not None]

    @property
    def adjoint(self) -> Wires:
        r"""A new Wires object with ket <-> bra."""
        w = self.copy()
        w._ids = {m: [w._ids[m][2], w._ids[m][3], w._ids[m][0], w._ids[m][1]] for m in w._ids}
        return w

    @property
    def dual(self) -> Wires:
        r"""A new Wires object with input <-> output."""
        w = self.copy()
        w._ids = {m: [w._ids[m][1], w._ids[m][0], w._ids[m][3], w._ids[m][2]] for m in w._ids}
        return w
