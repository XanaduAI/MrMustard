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

# pylint: disable=protected-access

__all__ = ["Wire", "Wires"]

Wire = int
r"""
An integer representing a wire in a tensor network.
TODO: Leaving this because it's used in other files for type-hinting,
    but let's decide whether we want this (and ``Mode``) or not.
"""


class Wires:
    r"""A class with wire functionality for tensor network applications.
    In MrMustard, ``CircuitComponent``s have a ``Wires`` object as attribute
    to handle the wires of the component and to connect components together.

    Wires are arranged into four groups, and each of the groups can
    span multiple modes:
                        _______________
    input bra modes --->|   circuit   |---> output bra modes
    input ket modes --->|  component  |---> output ket modes
                        ---------------

    The "standard order" mentioned below is output_bra for all modes,
    input_bra for all modes, output_ket for all modes, input_ket for all modes.
    We use this order when we inline the ids into a list, or when we reproduce the
    init args from the ids.

    A ``Wires`` object can return subsets (views) of itself. Available subsets are:

    - input/output  (wires on input/output side)
    - bra/ket       (wires on bra/ket side)
    - modes         (wires on the given modes)
    - id_subset     (wires with the given ids)

    For example, ``wires.input`` returns a ``Wires`` object with only the input wires
    (on bra and ket sides and on all the modes). Or ``wires.input.bra[(1,2)] returns a
    ``Wires`` object with only the input bra wires on modes 1 and 2.
    Note these are views of the original ``Wires`` object, i.e. we can set the ``ids``
    on the views and it will be set on the original, e.g. ``wires1.output.ids = wires2.input.ids``.

    ``Wires`` can also be added to one another, which returns a new ``Wires`` object with
    the wires of both objects combined (if there are duplicates, an error is raised).

    Args:
        modes_out_bra (Iterable[int]): The output modes on the bra side.
        modes_in_bra (Iterable[int]): The input modes on the bra side.
        modes_out_ket (Iterable[int]): The output modes on the ket side.
        modes_in_ket (Iterable[int]): The input modes on the ket side.

    Note that the order of the modes passed to initialize the object doesn't matter,
    as they get sorted at init time.
    """

    def __init__(
        self,
        modes_out_bra: Optional[Iterable[int]] = None,
        modes_in_bra: Optional[Iterable[int]] = None,
        modes_out_ket: Optional[Iterable[int]] = None,
        modes_in_ket: Optional[Iterable[int]] = None,
    ) -> None:
        modes_out_bra = modes_out_bra or []
        modes_in_bra = modes_in_bra or []
        modes_out_ket = modes_out_ket or []
        modes_in_ket = modes_in_ket or []

        self._modes = list(
            set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)
        )
        randint = settings.rng.integers  # MM random number generator
        pow = 50
        ob = {m: randint(1, 2**pow) if m in modes_out_bra else 0 for m in self._modes}
        ib = {m: randint(1, 2**pow) if m in modes_in_bra else 0 for m in self._modes}
        ok = {m: randint(1, 2**pow) if m in modes_out_ket else 0 for m in self._modes}
        ik = {m: randint(1, 2**pow) if m in modes_in_ket else 0 for m in self._modes}
        self._id_array = np.array([[ob[m], ib[m], ok[m], ik[m]] for m in self._modes])
        self._mask = np.ones_like(self._id_array)  # multiplicative mask

    def _args(self):
        r"returns the same args one needs to initialize this object."
        self._id_array[:, 0]
        ob_modes = np.array(self._modes)[self._id_array[:, 0] > 0].tolist()
        ib_modes = np.array(self._modes)[self._id_array[:, 1] > 0].tolist()
        ok_modes = np.array(self._modes)[self._id_array[:, 2] > 0].tolist()
        ik_modes = np.array(self._modes)[self._id_array[:, 3] > 0].tolist()
        return tuple(ob_modes), tuple(ib_modes), tuple(ok_modes), tuple(ik_modes)

    @classmethod
    def _from_data(cls, id_array=None, modes=None, mask=None):
        r"""Private class method to initialize Wires object from the given data."""
        w = cls()
        w._id_array = id_array if id_array is not None else w._id_array
        w._modes = modes if modes is not None else w._modes
        w._mask = mask if mask is not None else np.ones_like(w._id_array)
        return w

    def _view(self, masked_rows: tuple[int, ...] = (), masked_cols: tuple[int, ...] = ()) -> Wires:
        r"""A masked view of this Wires object."""
        w = self._from_data(self._id_array, self._modes, self._mask.copy())
        w._mask[masked_rows, :] = -1
        w._mask[:, masked_cols] = -1
        return w

    @property
    def id_array(self) -> np.ndarray:
        "The id_array of the available wires in the standard order (bra/ket x out/in x mode)."
        return self._id_array * self._mask

    @property
    def ids(self) -> list[int]:
        "The list of ids of the available wires in the standard order."
        flat = self.id_array.T.ravel()
        return flat[flat > 0].tolist()

    @ids.setter
    def ids(self, ids: list[int]):
        "Sets the ids of the available wires."
        if len(ids) != len(self.ids):
            raise ValueError(f"wrong number of ids (expected {len(self.ids)}, got {len(ids)})")
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
        return self._view(masked_cols=(0, 2))

    @property
    def output(self) -> Wires:
        "A view of this Wires object without input wires"
        return self._view(masked_cols=(1, 3))

    @property
    def ket(self) -> Wires:
        "A view of this Wires object without bra wires"
        return self._view(masked_cols=(0, 1))

    @property
    def bra(self) -> Wires:
        "A view of this Wires object without ket wires"
        return self._view(masked_cols=(2, 3))

    def adjoint(self) -> Wires:
        r"""
        The adjoint of this wires object, with new ids.
        """
        ob_modes, ib_modes, ok_modes, ik_modes = self._args()
        return Wires(ok_modes, ik_modes, ob_modes, ib_modes)

    def dual(self) -> Wires:
        r"""
        The dual of this wires object, with new ids.
        """
        ob_modes, ib_modes, ok_modes, ik_modes = self._args()
        return Wires(ib_modes, ob_modes, ik_modes, ok_modes)

    def copy(self) -> Wires:
        r"""A copy of this wire with new ids."""
        w = Wires(*self._args())
        w._mask = self._mask.copy()
        return w

    def subset(self, ids: Iterable[int]) -> Wires:
        "A subset of this Wires object with only the given ids."
        _id_array = np.where(np.isin(self._id_array, np.array(ids)), self._id_array, 0)
        modes = [self._modes[i] for i, row in enumerate(_id_array) if np.any(row != 0)]
        rows = [self._modes.index(m) for m in modes]
        return self._from_data(_id_array[rows], modes, self._mask[rows])

    def __add__(self, other: Wires) -> Wires:
        "A new Wires object with the wires of self and other combined."
        modes_rows = {}
        all_modes = sorted(set(self.modes) | set(other.modes))
        for m in all_modes:
            self_row = self.id_array[self._modes.index(m)] if m in self.modes else np.zeros(4)
            other_row = other.id_array[other._modes.index(m)] if m in other.modes else np.zeros(4)
            if np.any(np.where(self_row > 0) == np.where(other_row > 0)):
                raise ValueError(f"wires overlap on mode {m}")
            modes_rows[m] = [s if s > 0 else o for s, o in zip(self_row, other_row)]
        combined_array = [modes_rows[m] for m in sorted(modes_rows)]
        return self._from_data(combined_array, sorted(modes_rows), np.ones_like(combined_array))

    def __bool__(self) -> bool:
        return True if self.ids else False

    def __getitem__(self, modes: Iterable[int] | int) -> Wires:
        "A view of this Wires object with wires only on the given modes."
        modes = [modes] if isinstance(modes, int) else modes
        idxs = tuple(list(self._modes).index(m) for m in set(self._modes).difference(modes))
        return self._view(masked_rows=idxs)

    def __repr__(self) -> str:
        ob_modes, ib_modes, ok_modes, ik_modes = self._args()
        return f"Wires({ob_modes}, {ib_modes}, {ok_modes}, {ik_modes})"

    def _repr_html_(self):  # pragma: no cover
        "A matrix plot of the id_array."
        row_labels = map(str, self._modes)
        col_labels = ["bra-out", "bra-in", "ket-out", "ket-in"]
        array = np.abs(self.id_array) / (self.id_array + 1e-15)
        idxs = (i for i in self.indices)
        box_size = "60px"  # Set the size of the squares
        html = '<table style="border-collapse: collapse; border: 1px solid black;">'

        # colors
        active = "#5b9bd5"
        inactive = "#d6e8f7"

        # Add column headers
        html += "<tr>"
        for label in [""] + col_labels:  # Add an empty string for the top-left cell
            html += f'<th style="border: 1px solid black; padding: 5px;">{label}</th>'
        html += "</tr>"

        # Initialize rows with row labels
        rows_html = [
            f'<tr><td style="border: 1px solid black; padding: 5px;">{label}</td>'
            for label in row_labels
        ]

        # Add table cells (column by column)
        for label, col in zip(col_labels, array.T):
            for row_idx, value in enumerate(col):
                color = (
                    "white"
                    if np.isclose(value, 0)
                    else (active if np.isclose(value, 1) else inactive)
                )
                cell_html = f'<td style="border: 1px solid black; padding: 5px;\
                      width: {box_size}px; height: {box_size}px; background-color:\
                          {color}; box-sizing: border-box;'
                if color == active:
                    cell_html += (
                        f' text-align: center; vertical-align: middle;">{str(next(idxs))}</td>'
                    )
                else:
                    cell_html += '"></td>'
                rows_html[row_idx] += cell_html

        # Close the rows and add them to the HTML table
        for row_html in rows_html:
            row_html += "</tr>"
            html += row_html

        html += "</table>"

        try:
            from IPython.display import display, HTML  # pylint: disable=import-outside-toplevel

            display(HTML(html))
        except ImportError as e:
            raise ImportError(
                "To display the wires in a jupyter notebook you need to `pip install IPython`"
            ) from e

    def add_connected(self, other) -> Wires:
        "A new Wires object with the wires of self and other combined."
        all_modes = sorted(set(self.modes) | set(other.modes))

        ob = {
            m: 0 if m not in self.output.bra.modes else self.output.bra[m].ids[0] for m in all_modes
        }
        ib = {
            m: 0 if m not in self.input.bra.modes else self.input.bra[m].ids[0] for m in all_modes
        }
        ok = {
            m: 0 if m not in self.output.ket.modes else self.output.ket[m].ids[0] for m in all_modes
        }
        ik = {
            m: 0 if m not in self.input.ket.modes else self.input.ket[m].ids[0] for m in all_modes
        }

        msg = "Found wires that are not connected."
        for m in other.input.bra.modes:
            if ob[m]:
                if ob[m] == other.input.bra[m].ids[0]:
                    ob[m] = 0
                else:
                    raise ValueError(msg)
            else:
                ib[m] = other.input.bra[m].ids[0]
        for m in other.input.ket.modes:
            if ok[m]:
                if ok[m] == other.input.ket[m].ids[0]:
                    ok[m] = 0
                else:
                    raise ValueError(msg)
            else:
                ik[m] = other.input.ket[m].ids[0]

        for m in other.output.bra.modes:
            ob[m] = other.output.bra[m].ids[0]
        for m in other.output.ket.modes:
            ok[m] = other.output.ket[m].ids[0]

        combined_array = np.array([[ob[m], ib[m], ok[m], ik[m]] for m in all_modes])

        return self._from_data(combined_array, all_modes)
