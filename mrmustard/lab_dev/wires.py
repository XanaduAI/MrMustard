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
import numpy as np
from mrmustard import settings


class Wires:
    r"""A class with wire functionality for tensor network applications.
    Anything that wants wires should use an object of this class.
    For the time being, wires are tuples of type (int, Optional[int]), where
    the first int is the id of the wire of self, and the second int is the id
    of the wire the first is attached to, if any. The second int is None if
    the wire is not attached to anything.
    The Wires class is an orchestrator of the individual wires.

    Wires are arranged into four sets (each of the four sets can span multiple modes):

    input bra --->|   |---> output bra
    input ket --->|   |---> output ket
    
    A ``Wires`` object can return a subset ``Wires`` object. Available subsets are:
    - input/output
    - bra/ket
    - modes

    E.g. ``wires.input`` returns a Wires object with only the input wires
    (on bra/ket side and all modes). Or ``wires.bra[(1,2)] returns a Wires
    object with only the bra wires on modes 1 and 2 (on input/output side).

    Args:
        modes_out_bra (Iterable[int]): The output modes on the bra side.
        modes_in_bra (Iterable[int]): The input modes on the bra side.
        modes_out_ket (Iterable[int]): The output modes on the ket side.
        modes_in_ket (Iterable[int]): The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: Iterable[int] = set(),
        modes_in_bra: Iterable[int] = set(),
        modes_out_ket: Iterable[int] = set(),
        modes_in_ket: Iterable[int] = set(),
    ) -> None:
        self._modes = sorted(
            set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)
        )
        randint = settings.rng.integers  # MM random number generator
        ob = {m: randint(1, 2**62) if m in modes_out_bra else 0 for m in self._modes}
        ib = {m: randint(1, 2**62) if m in modes_in_bra else 0 for m in self._modes}
        ok = {m: randint(1, 2**62) if m in modes_out_ket else 0 for m in self._modes}
        ik = {m: randint(1, 2**62) if m in modes_in_ket else 0 for m in self._modes}
        self._id_array = np.array([[ob[m], ib[m], ok[m], ik[m]] for m in self._modes])
        self.mask = np.ones_like(self._id_array)  # multiplicative mask
        self.connections: dict[int, int] = {}

    def copy(self, id_array: Optional[np.ndarray] = None) -> Wires:
        "A disconnected soft copy of self with optional custom id_array."
        w = Wires(self.output.bra.modes, self.input.bra.modes, self.output.ket.modes,
                  self.input.ket.modes)
        w.mask = self.mask.copy()
        if id_array is not None:
            assert id_array.shape == self._id_array.shape, "incompatible id_array"
            w._id_array = id_array
        return w

    def view(self, masked_rows: list[int] = [], masked_cols: list[int] = []) -> Wires:
        r"""An masked view of this Wires object."""
        w = self.copy(self._id_array)
        w.mask[masked_rows, :] = -1
        w.mask[:, masked_cols] = -1
        w.connections = {id:self.connections[id] for id in w.ids if id in self.connections}  # subset of connections
        return w
    
    def __bool__(self) -> bool:
        "True if this Wires object contains any wires."
        return len(self.ids) > 0

    @property
    def id_array(self) -> np.ndarray:
        "The id_array of the available wires in the standard order (bra/ket x out/in x mode)."
        return self._id_array * self.mask
    
    @property
    def ids(self) -> list[int]:
        "The list of ids of the available wires in the standard order."
        flat = self.id_array.T.ravel()
        return flat[flat > 0].tolist()

    @property
    def modes(self) -> set[int]:
        "The set of modes of the populated wires."
        return set(m for m in self._modes if any(self.id_array[self._modes.index(m)] > 0))

    @property
    def indices(self) -> list[int]:
        r"""Returns the array of indices of this subset in the standard order.
        (bra/ket x out/in x mode). Use this to get the indices for bargmann contractions.
        """
        flat = self.id_array.T.ravel()
        flat = flat[flat != 0]
        return np.where(flat > 0)[0].tolist()

    @property
    def input(self) -> Wires:
        "A view of this Wires object without output wires"
        return self.view(masked_cols=[0, 2])

    @property
    def output(self) -> Wires:
        "A view of this Wires object without input wires"
        return self.view(masked_cols=[1, 3])

    @property
    def ket(self) -> Wires:
        "A view of this Wires object without bra wires"
        return self.view(masked_cols=[0, 1])

    @property
    def bra(self) -> Wires:
        "A view of this Wires object without ket wires"
        return self.view(masked_cols=[2, 3])

    def __getitem__(self, modes: Iterable[int] | int) -> Wires:
        "A view of this Wires object with wires only on the given modes."
        modes = [modes] if isinstance(modes, int) else modes
        idxs = [list(self._modes).index(m) for m in set(self._modes).difference(modes)]
        return self.view(masked_rows=idxs)

    @property
    def adjoint(self) -> Wires:
        "A new Wires object with ket <-> bra."
        w = self.copy(self._id_array[:, [1, 0, 3, 2]])
        w.mask = self.mask[:, [1, 0, 3, 2]]
        w.connections = {id:self.connections[id] for id in w.ids}
        return w

    @property
    def dual(self) -> Wires:
        "A new Wires object with input <-> output."
        w = self.copy(self._id_array[:, [2, 3, 0, 1]])
        w.mask = self.mask[:, [2, 3, 0, 1]]
        w.connections = {id:self.connections[id] for id in w.ids}
        return w
