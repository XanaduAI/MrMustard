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
from IPython.display import display, HTML
from mrmustard import settings

# pylint: disable=protected-access


class Wires:
    r"""A class with wire functionality for tensor network applications.
    Anything that wants wires should use an object of this class.

    Wires are arranged into four groups, and each of the four groups can span multiple modes.

    input bra --->|   |---> output bra
    input ket --->|   |---> output ket

    A ``Wires`` object can return sub-``Wires`` objects.
    Available subsets are:
    - input/output
    - bra/ket
    - modes

    E.g. ``wires.input`` returns a Wires object with only the input wires
    (on bra/ket side and all modes). Or ``wires.bra[(1,2)] returns a Wires
    object with only the bra wires on modes 1 and 2 (on input/output side).
    Note these are views of the original Wires object, i.e. if they are modified
    the original will change.

    Args:
        modes_out_bra (Iterable[int]): The output modes on the bra side.
        modes_in_bra (Iterable[int]): The input modes on the bra side.
        modes_out_ket (Iterable[int]): The output modes on the ket side.
        modes_in_ket (Iterable[int]): The input modes on the ket side.

    Note that the order of the modes passed to initialize the object doesn't matter,
    as they are sorted.
    """

    def __init__(
        self,
        modes_out_bra: Iterable[int] = (),
        modes_in_bra: Iterable[int] = (),
        modes_out_ket: Iterable[int] = (),
        modes_in_ket: Iterable[int] = (),
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

    @property
    def args(self):
        r"returns the same args one needs to initialize this object."
        ob_modes = np.array(self._modes)[self._id_array[:, 0] > 0].tolist()
        ib_modes = np.array(self._modes)[self._id_array[:, 1] > 0].tolist()
        ok_modes = np.array(self._modes)[self._id_array[:, 2] > 0].tolist()
        ik_modes = np.array(self._modes)[self._id_array[:, 3] > 0].tolist()
        return ob_modes, ib_modes, ok_modes, ik_modes

    def copy(self, id_array: Optional[np.ndarray] = None) -> Wires:
        r"""A copy of self with optional custom id_array. If id_array is passed,
        the copy will hold a reference to it (not a copy)."""
        w = Wires(*self.args)
        w.mask = self.mask.copy()
        if id_array is not None:
            assert id_array.shape == self._id_array.shape, "incompatible id_array"
            w._id_array = id_array
        return w

    def view(self, masked_rows: tuple[int, ...] = (), masked_cols: tuple[int, ...] = ()) -> Wires:
        r"""A masked view of this Wires object."""
        w = self.copy(self._id_array)
        w.mask[masked_rows, :] = -1
        w.mask[:, masked_cols] = -1
        return w

    def subset(self, ids: Iterable[int]) -> Wires:
        "A subset of this Wires object with only the given ids."
        subset = [self.ids.index(i) for i in ids if i in self.ids]
        w = Wires(self._id_array[subset])
        w.mask = self.mask[subset]
        w._modes = [self._modes[i] for i in subset]
        return w

    def __add__(self, other: Wires) -> Wires:
        "A new Wires object with the wires of self and other combined."
        modes_rows = {}
        all_modes = sorted(set(self.modes) | set(other.modes))
        for m in all_modes:
            self_row = self.id_array[self.modes.index(m)] if m in self.modes else np.zeros(4)
            other_row = other.id_array[other.modes.index(m)] if m in other.modes else np.zeros(4)
            assert np.all(np.where(self_row > 0) != np.where(other_row > 0)), "duplicate wires!"
            modes_rows[m] = [s if s > 0 else o for s, o in zip(self_row, other_row)]
        w = Wires()
        w._id_array = np.array([modes_rows[m] for m in sorted(modes_rows)])
        w.mask = np.ones_like(w._id_array)
        w._modes = sorted(modes_rows)
        return w

    @property
    def id_array(self) -> np.ndarray:
        "The id_array of the available wires in the standard order (bra/ket x out/in x mode)."
        return self._id_array * self.mask

    @property
    def ids(self) -> list[int]:
        "The list of ids of the available wires in the standard order."
        flat = self.id_array.T.ravel()
        return flat[flat > 0].tolist()

    @ids.setter
    def ids(self, ids: list[int]):
        "Sets the ids of the available wires."
        assert len(ids) == len(self.ids), "incompatible ids"
        self._id_array.flat[self.id_array.flatten() > 0] = ids

    @property
    def modes(self) -> list[int]:
        "The set of modes of the populated wires."
        return [m for m in self._modes if any(self.id_array[self._modes.index(m)] > 0)]

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

    def __repr__(self) -> str:
        ob_modes, ib_modes, ok_modes, ik_modes = self.args
        return f"Wires({ob_modes}, {ib_modes}, {ok_modes}, {ik_modes})"

    def _repr_html_(self):
        "A matrix plot of the id_array."
        data = np.abs(self.id_array) / (self.id_array + 1e-15)
        box_size = "60px"  # Set the size of the squares
        html = '<table style="border-collapse: collapse; border: 1px solid black;">'

        # Add column headers
        html += "<tr>"
        for col_label in ["", "bra-out", "bra-in", "ket-out", "ket-in"]:
            html += f'<th style="border: 1px solid black; padding: 5px;">{col_label}</th>'
        html += "</tr>"

        idxs = (i for i in self.indices)

        # Add rows
        for row_label, row in zip(self._modes, data):
            html += "<tr>"
            html += f'<td style="border: 1px solid black; padding: 5px;">{row_label}</td>'
            for value in row:
                color = (
                    "white" if np.isclose(value, 0) else ("red" if np.isclose(value, 1) else "pink")
                )
                html += f'<td style="border: 1px solid black; padding: 5px; width: {box_size}; height: {box_size}; background-color: {color}; box-sizing: border-box;'
                if color == "red":
                    html += f' text-align: center; vertical-align: middle; box-sizing: border-box;">{str(next(idxs))}</td>'
                else:
                    html += '"></td>'
            html += "</tr>"

        html += "</table>"
        display(HTML(html))
