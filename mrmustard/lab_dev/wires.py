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
from typing import Iterable, Optional
import uuid
import numpy as np


class Wires:
    r"""A class with wire functionality for tensor network applications.
    Anything that wants wires should use an object of this class.
    Note that the modes are sorted automatically.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: Iterable[int] = [],
        modes_in_bra: Iterable[int] = [],
        modes_out_ket: Iterable[int] = [],
        modes_in_ket: Iterable[int] = [],
    ) -> None:
        self._modes = (
            set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)
        )
        out_bra = {m: uuid.uuid4().int // 1e20 if m in modes_out_bra else 0 for m in self._modes}
        in_bra = {m: uuid.uuid4().int // 1e20 if m in modes_in_bra else 0 for m in self._modes}
        out_ket = {m: uuid.uuid4().int // 1e20 if m in modes_out_ket else 0 for m in self._modes}
        in_ket = {m: uuid.uuid4().int // 1e20 if m in modes_in_ket else 0 for m in self._modes}

        self._ids = np.array(
            [[out_bra[m], in_bra[m], out_ket[m], in_ket[m]] for m in self._modes], dtype=np.int64
        )
        self.mask = np.ones_like(self._ids, dtype=int)  # multiplicative mask

    @property
    def ids(self) -> np.ndarray:
        "The ids of the wires in the standard order (bra/ket x out/in x mode)."
        return self._ids * self.mask

    @property
    def modes(self) -> list[int]:
        "The modes of the available wires in the standard order."
        return [m for m in self._modes if any(self.ids[list(self._modes).index(m)] > 0)]

    def new(self, ids: Optional[np.ndarray] = None) -> Wires:
        "A copy of self with the given ids or new ids if ids is None."
        if ids is None:
            w = Wires(
                self.bra.output.modes,
                self.bra.input.modes,
                self.ket.output.modes,
                self.ket.input.modes,
            )
        else:
            w = Wires()
            w._modes = self._modes
            w._ids = ids
        w.mask = self.mask.copy()
        return w

    @property
    def indices(self) -> list[int]:
        r"""Returns the array of indices of the given id in the standard order.
        (bra/ket x out/in x mode). Use this to get the indices for bargmann contractions.
        """
        flat = self.ids.T.ravel()
        flat = flat[flat != 0]
        return np.where(flat > 0)[0].tolist()

    def masked_view(self, masked_rows=[], masked_cols=[]) -> Wires:
        r"""A view of this Wires object with the given mask."""
        w = self.new(self._ids)
        w.mask[masked_rows, :] = -1
        w.mask[:, masked_cols] = -1
        return w

    @property
    def input(self) -> Wires:
        "A view of this Wires object without output wires"
        return self.masked_view(masked_cols=[0, 2])

    @property
    def output(self) -> Wires:
        "A view of this Wires object without input wires"
        return self.masked_view(masked_cols=[1, 3])

    @property
    def ket(self) -> Wires:
        "A view of this Wires object without bra wires"
        return self.masked_view(masked_cols=[0, 1])

    @property
    def bra(self) -> Wires:
        "A view of this Wires object without ket wires"
        return self.masked_view(masked_cols=[2, 3])

    def __getitem__(self, modes: Iterable[int] | int) -> Wires:
        "A view of this Wires object with wires only on the given modes."
        modes = [modes] if isinstance(modes, int) else modes
        idxs = [list(self._modes).index(m) for m in self._modes.difference(modes)]
        return self.masked_view(masked_rows=idxs)

    @property
    def adjoint(self) -> Wires:
        "A new Wires object with ket <-> bra."
        w = self.new(self._ids[:, [1, 0, 3, 2]])
        w.mask = self.mask[:, [1, 0, 3, 2]]
        return w

    @property
    def dual(self) -> Wires:
        "A new Wires object with input <-> output."
        w = self.new(self._ids[:, [2, 3, 0, 1]])
        w.mask = self.mask[:, [2, 3, 0, 1]]
        return w
