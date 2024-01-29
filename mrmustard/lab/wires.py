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

"""`Wires` class for handling the connectivity of an object in a circuit."""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
from mrmustard import settings

# pylint: disable=protected-access


class Wires:
    r"""`Wires` class for handling the connectivity of an object in a circuit.
    In MrMustard, ``CircuitComponent``s have a ``Wires`` object as attribute
    to handle the wires of the component and to connect components together.

    Wires are arranged into four groups, and each of the groups can
    span multiple modes:
                        _______________
    input bra modes --->|   circuit   |---> output bra modes
    input ket modes --->|  component  |---> output ket modes
                        ---------------
    Each of the four groups can be empty. In particular, the wires of a state have
    no inputs and the wires of a measurement have no outputs. Similarly,
    a standard unitary transformation has no bra wires.

    We refer to these four groups in a "standard order":
        0. output_bra for all modes
        1. input_bra for all modes
        2. output_ket for all modes
        3. input_ket for all modes

    A ``Wires`` object can return subsets (views) of itself. Available subsets are:
    - input/output  (wires on input/output side)
    - bra/ket       (wires on bra/ket side)
    - modes         (wires on the given modes)

    For example, ``wires.input`` returns a ``Wires`` object with only the input wires
    (on bra and ket sides and on all the modes).
    Note that these can be combined together: ``wires.input.bra[(1,2)] returns a
    ``Wires`` object with only the input bra wires on modes 1 and 2.
    Note these are views of the original ``Wires`` object, i.e. we can set the ``ids`` on the
    views and it will be set on the original, e.g. this is a valid way to connect two sets
    of wires by setting their ids to be equal: ``wires1.output.ids = wires2.input.ids``.

    A very useful feature of the ``Wires`` class is the support for the right shift
    operator ``>>``. This allows us to connect two ``Wires`` objects together as
    ``wires1 >> wires2``. This will return the ``Wires`` object of the two
    wires objects connected together as if they were in a circuit:
               ____________                             ____________
    in bra --->|  wires1  |---> out bra ---> in bra --->|  wires2  |---> out bra
    in ket --->|          |---> out ket ---> in ket --->|          |---> out ket
               ------------                             ------------
    The returned ``Wires`` object will contain the surviving wires of the two
    ``Wires`` objects and it will raise an error if there are overlaps between
    the surviving wires. This is especially useful for handling the ordering of the
    wires when connecting components together: we are always guaranteed that a
    ``Wires`` object will provide the wires in the standard order.

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

        self._modes = sorted(
            set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)
        )
        randint = settings.rng.integers  # MM random number generator
        outbra = {m: randint(1, 2**62) if m in modes_out_bra else 0 for m in self._modes}
        inbra = {m: randint(1, 2**62) if m in modes_in_bra else 0 for m in self._modes}
        outket = {m: randint(1, 2**62) if m in modes_out_ket else 0 for m in self._modes}
        inket = {m: randint(1, 2**62) if m in modes_in_ket else 0 for m in self._modes}
        self._id_array = np.array([[outbra[m], inbra[m], outket[m], inket[m]] for m in self._modes])
        self._mask = np.ones_like(self._id_array)  # multiplicative mask

    def _args(self):
        r"""Returns the same args one needs to initialize this object."""
        ob_modes = np.array(self._modes)[self._id_array[:, 0] > 0].tolist()
        ib_modes = np.array(self._modes)[self._id_array[:, 1] > 0].tolist()
        ok_modes = np.array(self._modes)[self._id_array[:, 2] > 0].tolist()
        ik_modes = np.array(self._modes)[self._id_array[:, 3] > 0].tolist()
        return tuple(ob_modes), tuple(ib_modes), tuple(ok_modes), tuple(ik_modes)

    @classmethod
    def _from_data(cls, id_array: np.ndarray, modes: list[int], mask=None):
        r"""Private class method to initialize Wires object from the given data."""
        w = cls()
        w._id_array = id_array
        w._modes = modes
        w._mask = mask if mask is not None else np.ones_like(w._id_array)
        return w

    def _view(self, masked_rows: tuple[int, ...] = (), masked_cols: tuple[int, ...] = ()) -> Wires:
        r"""A masked view of this Wires object."""
        w = self._from_data(self._id_array, self._modes, self._mask.copy())
        w._mask[masked_rows, :] = -1
        w._mask[:, masked_cols] = -1
        return w

    def _mode(self, mode: int) -> np.ndarray:
        "A slice of the id_array matrix at the given mode."
        return np.maximum(0, self.id_array[[self._modes.index(mode)]])[0]

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
        "The set of modes spanned by the populated wires."
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
        "A view of self without output wires"
        return self._view(masked_cols=(0, 2))

    @property
    def output(self) -> Wires:
        "A view of self without input wires"
        return self._view(masked_cols=(1, 3))

    @property
    def ket(self) -> Wires:
        "A view of self without bra wires"
        return self._view(masked_cols=(0, 1))

    @property
    def bra(self) -> Wires:
        "A view of self without ket wires"
        return self._view(masked_cols=(2, 3))

    @property
    def adjoint(self) -> Wires:
        r"""
        The adjoint (ket <-> bra) of this wires object.
        """
        return self._from_data(self._id_array[:, [2, 3, 0, 1]], self._modes, self._mask)

    @property
    def dual(self) -> Wires:
        r"""
        The dual (in <-> out) of this wires object.
        """
        return self._from_data(self._id_array[:, [1, 0, 3, 2]], self._modes, self._mask)

    def copy(self) -> Wires:
        r"""A copy of this Wires object with new ids."""
        w = Wires(*self._args())
        w._mask = self._mask.copy()
        return w

    def __bool__(self) -> bool:
        return len(self.ids) > 0

    def __getitem__(self, modes: Iterable[int] | int) -> Wires:
        "A view of this Wires object with wires only on the given modes."
        modes = [modes] if isinstance(modes, int) else modes
        idxs = tuple(list(self._modes).index(m) for m in set(self._modes).difference(modes))
        return self._view(masked_rows=idxs)

    def __lshift__(self, other: Wires) -> Wires:
        return (other.dual >> self.dual).dual  # how cool is this

    @staticmethod
    def _outin(si, so, oi, oo):
        r"""Returns the output and input wires of the composite object made by connecting
        two single-mode ket (or bra) objects like --|self|-- and --|other|--
        At this stage we are guaranteed that the configurations `|self|--  |other|--`  and 
        `--|self|  --|other|` (which would be invalid) have already been excluded.
        """
        if bool(so) == bool(oi):  # if the inner wires are either both there or both not there
            return np.array([oo, si], dtype=np.int64)
        elif not si and not so:  # no wires on self
            return np.array([oo, oi], dtype=np.int64)
        else:  # no wires on other
            return np.array([so, si], dtype=np.int64)

    def __rshift__(self, other: Wires) -> Wires:
        r"""Returns a new Wires object with the wires of self and other combined as two
        components in a circuit where the output of self connects to the input of other:
            ``self >> other``
        All surviving wires are arranged in the standard order.
        A ValueError is raised if there are any surviving wires that overlap, which is the only
        possible way two objects aren't compatible in a circuit."""
        all_modes = sorted(set(self.modes) | set(other.modes))
        new_id_array = np.zeros((len(all_modes), 4), dtype=np.int64)
        for i, m in enumerate(all_modes):
            if m in self.modes and m in other.modes:
                sob, sib, sok, sik = self._mode(m)  # m-th row of self
                oob, oib, ook, oik = other._mode(m)  # m-th row of other
                errors = {
                    "output bra": sob and oob and not oib, #  |s|- |o|- (bra)
                    "output ket": sok and ook and not oik, #  |s|- |o|- (ket)
                    "input bra": oib and sib and not sob,  # -|s| -|o|  (bra)
                    "input ket": oik and sik and not sok,  # -|s| -|o|  (ket)
                }
                if any(errors.values()):
                    position = [k for k, v in errors.items() if v][0]
                    raise ValueError(f"{position} wire overlap at mode {m}")
                new_id_array[i] += np.hstack([self._outin(sib, sob, oib, oob), self._outin(sik, sok, oik, ook)])
            elif m in self.modes and m not in other.modes:
                new_id_array[i] += self._mode(m)
            elif m in other.modes and m not in self.modes:
                new_id_array[i] += other._mode(m)
        return self._from_data(new_id_array, all_modes)  # abs to turn hidden ids (negative) into visible

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
            from IPython.core.display import display, HTML  # pylint: disable=import-outside-toplevel
            display(HTML(html))
        except ImportError as e:
            raise ImportError(
                "To display the wires in a jupyter notebook you need to `pip install IPython` first."
            ) from e
