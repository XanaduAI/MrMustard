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

""" ``Wires`` class for supporting tensor network functionalities."""

from __future__ import annotations
from collections import defaultdict

from typing import Iterable, Collection, Optional
from IPython.display import display, HTML
import numpy as np

from mrmustard import settings

__all__ = ["Wires"]

    # def _repr_html_(self):  # pragma: no cover
    #     "A matrix plot of the id_array."
    #     row_labels = map(str, self._modes)
    #     col_labels = ["bra-out", "bra-in", "ket-out", "ket-in"]
    #     array = np.abs(self.id_array) / (self.id_array + 1e-15)
    #     idxs = (i for i in self.indices)
    #     box_size = "60px"  # Set the size of the squares
    #     html = '<table style="border-collapse: collapse; border: 1px solid black;">'
    #     # colors
    #     active = "#5b9bd5"
    #     inactive = "#d6e8f7"

    #     # Add column headers
    #     html += "<tr>"
    #     for label in [""] + col_labels:  # Add an empty string for the top-left cell
    #         html += f'<th style="border: 1px solid black; padding: 5px;">{label}</th>'
    #     html += "</tr>"

    #     # Initialize rows with row labels
    #     rows_html = [
    #         f'<tr><td style="border: 1px solid black; padding: 5px;">{label}</td>'
    #         for label in row_labels
    #     ]

    #     # Add table cells (column by column)
    #     for label, col in zip(col_labels, array.T):
    #         for row_idx, value in enumerate(col):
    #             color = (
    #                 "white"
    #                 if np.isclose(value, 0)
    #                 else (active if np.isclose(value, 1) else inactive)
    #             )
    #             cell_html = f'<td style="border: 1px solid black; padding: 5px;\
    #                   width: {box_size}px; height: {box_size}px; background-color:\
    #                       {color}; box-sizing: border-box;'
    #             if color == active:
    #                 cell_html += (
    #                     f' text-align: center; vertical-align: middle;">{str(next(idxs))}</td>'
    #                 )
    #             else:
    #                 cell_html += '"></td>'
    #             rows_html[row_idx] += cell_html

    #     # Close the rows and add them to the HTML table
    #     for row_html in rows_html:
    #         row_html += "</tr>"
    #         html += row_html

    #     html += "</table>"
    #     display(HTML(html))


# pylint: disable=protected-access
class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, we represent circuit components as tensors in a tensor network. The wires of
    these components describe how they connect with the surrounding components. For example, an
    `N`-mode pure state has `N` ket wires on the output side, while a `N`-mode mixed state
    has `N` ket wires and `N` bra wires on the output side.

    ``Wires`` objects store the information related to the wires of circuit components. Each wire
    in a ``Wires`` object is specified by a numerical id, which is random and unique. When two different
    ``Wires`` object have one or more wires with the same ids, we treat them as connected. Otherwise,
    we treat them as disconnected.

    The list of all these ids can be accessed using the ``ids`` property.

    .. code-block::

        >>> from mrmustard.lab_dev.wires import Wires

        >>> modes_out_bra=[0, 1]
        >>> modes_in_bra=[1, 2]
        >>> modes_out_ket=[0, 3]
        >>> modes_in_ket=[1, 2, 3]
        >>> w = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)

        >>> # access the modes
        >>> modes = w.modes
        >>> assert w.modes == [0, 1, 2, 3]

        >>> # access the ids
        >>> ids = w.ids
        >>> assert len(ids) == 9

        >>> # get input/output subsets
        >>> w_in = w.input
        >>> assert w_in.modes == [1, 2, 3]

        >>> # get ket/bra subsets
        >>> w_in_bra = w_in.bra
        >>> assert w_in_bra.modes == [1, 2]

    The standard order for the list of ids is:

    - ids for all the output bra wires.

    - ids for all the input bra wires.

    - ids for all the output ket wires.

    - ids for all the input ket wires.

    .. code-block::

        >>> assert w.output.bra.ids == w.ids[:2]
        >>> assert w.input.bra.ids == w.ids[2:4]
        >>> assert w.output.ket.ids == w.ids[4:6]
        >>> assert w.input.ket.ids == w.ids[6:]

    To access the index of a su set of wires in standard order (i.e. skipping over wires not belonging to the subset),
    one can use the ``indices`` attribute:

    .. code-block::

        >>> w = Wires(modes_in_ket = [0,1], modes_out_ket = [0,1])

        >>> assert w.indices == [0,1,2,3]
        >>> assert w.input.indices == [2,3]

    Note that subsets return new ``Wires`` objects with the same ids as the original object.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.

    Note that the order of the modes passed to initialize the object doesn't matter,
    as they get sorted at init time.
    """

    def __init__(
        self,
        modes_out_bra: Collection[int] = (),
        modes_in_bra: Collection[int] = (),
        modes_out_ket: Collection[int] = (),
        modes_in_ket: Collection[int] = (),
    ) -> None:

        self.types = (0,) * bool(modes_out_bra) + (1,) * bool(modes_in_bra) + (2,) * bool(modes_out_ket) + (3,) * bool(modes_in_ket)
        self._modes = tuple(sorted(set(modes_out_bra) | set(modes_in_bra) | set(modes_out_ket) | set(modes_in_ket)))
        ids = settings.rng.integers(1, 2**62, 4*len(self._modes)).tolist()
        self.data: dict[int,dict[int,tuple[int,int]]] = defaultdict(dict)
        for i,m in enumerate(modes_out_bra):
            self.data[0][m] = (i,ids.pop(0))
        for i,m in enumerate(modes_in_bra):
            self.data[1][m] = (i+len(modes_out_bra),ids.pop(0))
        for i,m in enumerate(modes_out_ket):
            self.data[2][m] = (i+len(modes_out_bra)+len(modes_in_bra),ids.pop(0))
        for i,m in enumerate(modes_in_ket):
            self.data[3][m] = (i+len(modes_out_bra)+len(modes_in_bra)+len(modes_out_ket),ids.pop(0))

    def view(self, types: Optional[tuple] = None, modes: Optional[tuple] = None):
        r"""
        A view of this Wires object with different order and modes.

        Args:
            types (tuple[int,...]): The types of the wires. always length 4.
            modes (tuple[int,...]): The modes of the wires.
        """
        w = Wires()
        w.data = self.data
        w.types = types if types is not None else self.types
        w._modes = modes if modes is not None else self._modes
        return w
    
    @property
    def modes(self) -> tuple[int,...]:
        r"""
        The modes of the wires in the standard order.
        """
        return tuple(sorted(set().union(*self.args)))

    @property
    def args(self) -> tuple[tuple[int,...],...]:
        r"""
        Returns the input arguments needed to initialize the same ``Wires`` object,
        """
        return tuple(tuple(m for m in self.data[t] if m in self._modes) if t in self.types else () for t in (0,1,2,3))

    @property
    def indices(self) -> tuple[int,...]:
        r"""
        The array of indices of this ``Wires`` in the standard order. The array of indices
        of this ``Wires`` in the standard order. When a subset is selected, it skips the
        indices of wires that do not belong to the subset.

        .. code-block::

            >>> w = Wires(modes_in_ket = (0,1), modes_out_ket = (0,1))

            >>> assert w.indices == (0,1,2,3)
            >>> assert w.input.indices == (2,3)
        """
        return tuple(self.data[t][m][0] for t in self.types for m in self.modes if m in self.data[t])
    
    @property
    def ids(self) -> tuple[int,...]:
        r"""
        The tuple of ids of the available wires in the standard order.
        """
        return tuple(self.data[t][m][1] for t in self.types for m in self.modes if m in self.data[t])

    @property
    def input(self) -> Wires:
        r"""
        A view of this ``Wires`` object without output wires.
        """
        return self.view(types = tuple(t for t in self.types if t not in (0,2)))

    @property
    def output(self) -> Wires:
        r"""
        A view of this ``Wires`` object without input wires.
        """
        return self.view(types = tuple(t for t in self.types if t not in (1,3)))

    @property
    def ket(self) -> Wires:
        r"""
        A view of this ``Wires`` object without bra wires.
        """
        return self.view(types = tuple(t for t in self.types if t not in (0,1)))

    @property
    def bra(self) -> Wires:
        r"""
        A view of this ``Wires`` object without ket wires.
        """
        return self.view(types = tuple(t for t in self.types if t not in (2,3)))

    @property
    def adjoint(self) -> Wires:
        r"""
        A new ``Wires`` object obtained by swapping ket and bra wires.
        """
        return Wires(self.data[2], self.data[3], self.data[0], self.data[1])

    @property
    def dual(self) -> Wires:
        r"""
        A new ``Wires`` object obtained by swapping input and output wires.
        """
        return Wires(self.data[1], self.data[0], self.data[3], self.data[2])
    
    def __getitem__(self, modes: tuple[int,...] | int) -> Wires:
        r"""
        A view of this Wires object with wires only on the given modes.
        """
        return self.view(modes = (modes,) if isinstance(modes,int) else tuple(modes))

    def __add__(self, other: Wires) -> Wires:
        r"""
        A new ``Wires`` object that combines the wires of ``self`` and those of ``other``.
        Raises:
            ValueError: If any leftover wires would overlap.
        """
        new_args = []
        for m1, m2 in zip(self.args, other.args):
            if set(m1) & set(m2):
                raise ValueError(f"wires overlap on mode(s) {set(m1) & set(m2)}")
            new_args.append(sorted(m1 + m2))
        return Wires(*new_args)

    def __bool__(self) -> bool:
        r"""
        Returns ``True`` if this ``Wires`` object has wires, ``False`` otherwise.
        """
        return len(self.indices) > 0

    def __eq__(self, other) -> bool:
        return self.args == other.args

    def __matmul__(self, other: Wires) -> Wires:
        r"""
        A new ``Wires`` object with the wires of ``self`` and ``other`` combined.

        The output of ``self`` connects to the input of ``other`` wherever they match. All
        surviving wires are arranged in the standard order.

        This function does not add missing adjoints.

        Raises:
            ValueError: If there are any surviving wires that overlap.
        """
        all_modes = sorted(set(self.modes) | set(other.modes))
        # bra
        self_in = np.array([m in self.data[1] for m in all_modes], dtype=np.int8)
        self_out = np.array([m in self.data[0] for m in all_modes], dtype=np.int8)
        other_in = np.array([m in other.data[1] for m in all_modes], dtype=np.int8)
        other_out = np.array([m in other.data[0] for m in all_modes], dtype=np.int8)
        bra_in = self_in + (other_in - self_out) * other_in
        bra_out = other_out + (self_out - other_in) * self_out
        if 2 in bra_in or 2 in bra_out:
            raise ValueError("bra wires overlap")
        # ket
        self_in = np.array([m in self.data[3] for m in all_modes], dtype=np.int8)
        self_out = np.array([m in self.data[2] for m in all_modes], dtype=np.int8)
        other_in = np.array([m in other.data[3] for m in all_modes], dtype=np.int8)
        other_out = np.array([m in other.data[2] for m in all_modes], dtype=np.int8)
        ket_in = self_in + (other_in - self_out) * other_in
        ket_out = other_out + (self_out - other_in) * self_out
        if 2 in ket_in or 2 in ket_out:
            raise ValueError("ket wires overlap")
        modes_bra_out = [m for i,m in enumerate(all_modes) if bra_out[i] == 1]
        modes_bra_in = [m for i,m in enumerate(all_modes) if bra_in[i] == 1]
        modes_ket_out = [m for i,m in enumerate(all_modes) if ket_out[i] == 1]
        modes_ket_in = [m for i,m in enumerate(all_modes) if ket_in[i] == 1]
        return Wires(modes_bra_out, modes_bra_in, modes_ket_out, modes_ket_in)

    def __repr__(self) -> str:
        return f"Wires({self.args})"