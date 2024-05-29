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

"""``Wires`` class for supporting tensor network functionalities."""

from __future__ import annotations
from functools import cached_property
from typing import Optional
import os
import numpy as np

from IPython.display import display, HTML
from mako.template import Template

__all__ = ["Wires"]


class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, instances of ``CircuitComponent`` have a ``Wires`` attribute.
    The wires describe how they connect with the surrounding components in a tensor network picture,
    where states flow from left to right. ``CircuitComponent``\s can have wires on the
    bra and/or on the ket side. Here are some examples for the types of components available on
    ``mrmustard.lab_dev``:

    .. code-block::

        A channel acting on mode ``1`` has input and output wires on both ket and bra sides:

        ┌──────┐   1  ╔═════════╗   1  ┌───────┐
        │Bra in│─────▶║         ║─────▶│Bra out│
        └──────┘      ║ Channel ║      └───────┘
        ┌──────┐   1  ║         ║   1  ┌───────┐
        │Ket in│─────▶║         ║─────▶│Ket out│
        └──────┘      ╚═════════╝      └───────┘


        A unitary acting on mode ``2`` has input and output wires on the ket side:

        ┌──────┐   2  ╔═════════╗   2  ┌───────┐
        │Ket in│─────▶║ Unitary ║─────▶│Ket out│
        └──────┘      ╚═════════╝      └───────┘


        A density matrix representing the state of mode ``0`` has only output wires:

                        ╔═════════╗   0  ┌───────┐
                        ║         ║─────▶│Bra out│
                        ║ Density ║      └───────┘
                        ║ Matrix  ║   0  ┌───────┐
                        ║         ║─────▶│Ket out│
                        ╚═════════╝      └───────┘


        Also a ket representing the state of mode ``1`` has only output wires:

                        ╔═════════╗   1  ┌───────┐
                        ║   Ket   ║─────▶│Ket out│
                        ╚═════════╝      └───────┘

    The ``Wires`` class can then be used to create subsets of wires:

    .. code-block::

        >>> from mrmustard.lab_dev.wires import Wires

        >>> modes_out_bra={0, 1}
        >>> modes_in_bra={1, 2}
        >>> modes_out_ket={0, 13}
        >>> modes_in_ket={1, 2, 13}
        >>> w = Wires(modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket)

        >>> # all the modes
        >>> modes = w.modes
        >>> assert w.modes == {0, 1, 2, 13}

        >>> # input/output modes
        >>> assert w.input.modes == {1, 2, 13}
        >>> assert w.output.modes == {0, 1, 13}

        >>> # get ket/bra modes
        >>> assert w.ket.modes == {0, 1, 2, 13}
        >>> assert w.bra.modes == {0, 1, 2}

        >>> # combined subsets
        >>> assert w.output.ket.modes == {0, 13}
        >>> assert w.input.bra.modes == {1, 2}

    Here's a diagram of the original ``Wires`` object in the example above,
    with the indices of the wires (the number in parenthesis) given in the "standard" order
    (``bra_out``, ``bra_in``, ``ket_out``, ``ket_in``, and the modes in sorted increasing order):

    .. code-block::

                     ╔═════════╗
        1 (2) ─────▶ ║         ║─────▶ 0 (0)
        2 (3) ─────▶ ║         ║─────▶ 1 (1)
                     ║         ║
                     ║  ``Wires``  ║
        1 (6) ─────▶ ║         ║
        2 (7) ─────▶ ║         ║─────▶ 0 (4)
       13 (8) ─────▶ ║         ║─────▶ 13 (5)
                     ╚═════════╝

    To access the index of a subset of wires in standard order we can use the ``indices``
    property:

    .. code-block::

        >>> assert w.indices == (0,1,2,3,4,5,6,7,8)
        >>> assert w.input.indices == (2,3,6,7,8)

    Another important application of the ``Wires`` class is to contract the wires of two components.
    This is done using the ``@`` operator. The result is a new ``Wires`` object that combines the wires
    of the two components. Here's an example of a contraction of a single-mode density matrix going
    into a single-mode channel:

    .. code-block::

        >>> rho = Wires(modes_out_bra={0}, modes_in_bra={0})
        >>> Phi = Wires(modes_out_bra={0}, modes_in_bra={0}, modes_out_ket={0}, modes_in_ket={0})
        >>> rho_out, perm = rho @ Phi
        >>> assert rho_out.modes == {0}

    Here's a diagram of the result of the contraction:

    .. code-block::

        ╔═══════╗      ╔═══════╗
        ║       ║─────▶║       ║─────▶ 0
        ║  rho  ║      ║  Phi  ║
        ║       ║─────▶║       ║─────▶ 0
        ╚═══════╝      ╚═══════╝

    The permutation that takes the contracted representations to the standard order is also returned.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: Optional[set[int]] = None,
        modes_in_bra: Optional[set[int]] = None,
        modes_out_ket: Optional[set[int]] = None,
        modes_in_ket: Optional[set[int]] = None,
    ) -> None:
        self.args: tuple[set, ...] = (
            modes_out_bra or set(),
            modes_in_bra or set(),
            modes_out_ket or set(),
            modes_in_ket or set(),
        )

        # The "parent" wires object, if any. This is ``None`` for freshly initialized
        # wires objects, and ``not None`` for subsets.
        self._original = None

        # Adds elements to the cache when calling ``__getitem__``
        self._mode_cache = {}

    def __len__(self) -> int:
        r"The number of wires."
        return sum(len(s) for s in self.args)

    @cached_property
    def id(self) -> int:
        r"""
        A numerical identifier for this ``Wires`` object.

        The ``id`` are random and unique, and are preserved when taking subsets.
        """
        if self.original:
            return self.original.id
        return np.random.randint(0, 2**31)

    @cached_property
    def ids(self) -> list[int]:
        r"""
        A list of numerical identifier for the wires in this ``Wires`` object, in
        the standard order.

        The ``ids`` are derived incrementally from the ``id`` and are unique.

        .. code-block::

            >>> w = Wires(modes_in_ket = {0,1}, modes_out_ket = {0,1})
            >>> id = w.id
            >>> ids = w.ids
            >>> assert ids == [id, id+1, id+2, id+3]
        """
        if self.original:
            return [self.original.ids[i] for i in self.indices]
        return [id for d in self.ids_dicts for id in d.values()]

    @cached_property
    def indices(self) -> tuple[int, ...]:
        r"""
        The array of indices of this ``Wires`` in the standard order.
        When a subset is selected (e.g. ``.ket``), it doesn't include wires that do not belong
        to the subset, but it still counts them because indices refer to the original modes.

        .. code-block::

            >>> w = Wires(modes_in_ket = {0,1}, modes_out_ket = {0,1})
            >>> assert w.indices == (0,1,2,3)
            >>> assert w.input.indices == (2,3)
        """
        return tuple(
            self.index_dicts[t][m]
            for t, modes in enumerate(self.sorted_args)
            for m in modes
        )

    @cached_property
    def index_dicts(self) -> list[dict[int, int]]:
        r"""
        A list of dictionary mapping modes to indices, one for each of the subsets
        (``output.bra``, ``input.bra``, ``output.ket``, and ``input.ket``).

        If subsets are taken, ``index_dicts`` refers to the parent object rather than to the
        child.
        """
        if self.original:
            return self.original.index_dicts
        return [
            {m: i + sum(len(s) for s in self.args[:t]) for i, m in enumerate(lst)}
            for t, lst in enumerate(self.sorted_args)
        ]

    @cached_property
    def ids_dicts(self) -> list[dict[int, int]]:
        r"""
        A list of dictionary mapping modes to ``ids``, one for each of the subsets
        (``output.bra``, ``input.bra``, ``output.ket``, and ``input.ket``).

        If subsets are taken, ``ids_dicts`` refers to the parent object rather than to the
        child.
        """
        if self.original:
            return self.original.ids_dicts
        return [{m: i + self.id for m, i in d.items()} for d in self.index_dicts]

    @cached_property
    def sorted_args(self) -> tuple[list[int], ...]:
        r"The sorted arguments. Allows to sort them only once."
        return tuple(sorted(s) for s in self.args)

    @property
    def original(self):
        r"""
        The parent wire, if any.
        """
        return self._original

    @cached_property
    def modes(self) -> set[int]:
        r"The modes spanned by the wires."
        return set.union(*self.args)

    @cached_property
    def input(self) -> Wires:
        r"New ``Wires`` object without output wires."
        ret = Wires(set(), self.args[1], set(), self.args[3])
        ret._original = self.original or self  # pylint: disable=protected-access
        return ret

    @cached_property
    def output(self) -> Wires:
        r"New ``Wires`` object without input wires."
        ret = Wires(self.args[0], set(), self.args[2], set())
        ret._original = self.original or self  # pylint: disable=protected-access
        return ret

    @cached_property
    def ket(self) -> Wires:
        r"New ``Wires`` object without bra wires."
        ret = Wires(set(), set(), self.args[2], self.args[3])
        ret._original = self.original or self  # pylint: disable=protected-access
        return ret

    @cached_property
    def bra(self) -> Wires:
        r"New ``Wires`` object without ket wires."
        ret = Wires(self.args[0], self.args[1], set(), set())
        ret._original = self.original or self  # pylint: disable=protected-access
        return ret

    @cached_property
    def adjoint(self) -> Wires:
        r"New ``Wires`` object obtained by swapping ket and bra wires."
        return Wires(self.args[2], self.args[3], self.args[0], self.args[1])

    @cached_property
    def dual(self) -> Wires:
        r"New ``Wires`` object obtained by swapping input and output wires."
        return Wires(self.args[1], self.args[0], self.args[3], self.args[2])

    def __getitem__(self, modes: tuple[int, ...] | int) -> Wires:
        r"New ``Wires`` object with wires only on the given modes."
        modes_set = {modes} if isinstance(modes, int) else set(modes)
        if modes not in self._mode_cache:
            w = Wires(*(self.args[t] & modes_set for t in (0, 1, 2, 3)))
            w._original = self.original or self
            self._mode_cache[modes] = w
        return self._mode_cache[modes]

    def __add__(self, other: Wires) -> Wires:
        r"""
        New ``Wires`` object that combines the wires of ``self`` and those of ``other``.
        Raises:
            ValueError: If any leftover wires would overlap.
        """
        new_args = []
        for t, (m1, m2) in enumerate(zip(self.args, other.args)):
            if m := m1 & m2:
                raise ValueError(f"{t}-type wires overlap at mode {m}")
            new_args.append(m1 | m2)
        return Wires(*new_args)

    def __bool__(self) -> bool:
        r"Returns ``True`` if this ``Wires`` object has any wires, ``False`` otherwise."
        return any(self.args)

    def __eq__(self, other) -> bool:
        return self.args == other.args

    def __matmul__(self, other: Wires) -> tuple[Wires, list[int]]:
        r"""
        Returns the wires of the circuit composition of self and other without adding missing
        adjoints. It also returns the permutation that takes the contracted representations
        to the standard order. An exception is raised if any leftover wires would overlap.
        Consider the following example:

        .. code-block::

                ╔═══════╗           ╔═══════╗
            B───║ self  ║───A   D───║ other ║───C
            b───║       ║───a   d───║       ║───c
                ╚═══════╝           ╚═══════╝

        B and D-A must not overlap, same for b and d-a, etc. The result is a new ``Wires`` object

        .. code-block::

                       ╔═══════╗
            B|(D-A)────║self @ ║────C|(A-D)
            b|(d-a)────║ other ║────c|(a-d)
                       ╚═══════╝

        In comparison, contracting the representations rather than the wires corresponds to
        an order where we start from juxtaposing the objects and then removing pairs of contracted
        indices, i.e. A-D, B, C, D-A and then the same for a-d, b, c, d-a. The returned permutation
        is the one that takes the result of multiplying representations to the standard order.

        Args:
            other: The wires of the other circuit component.

        Returns:
            The wires of the circuit composition and the permutation.

        Raises:
            ValueError: If any leftover wires would overlap.
        """
        if self.original or other.original:
            raise ValueError("cannot contract a subset of wires")
        A, B, a, b = self.args
        C, D, c, d = other.args
        sets = (A - D, B, a - d, b, C, D - A, c, d - a)
        if m := sets[0] & sets[4]:
            raise ValueError(f"output bra modes {m} overlap")
        if m := sets[1] & sets[5]:
            raise ValueError(f"input bra modes {m} overlap")
        if m := sets[2] & sets[6]:
            raise ValueError(f"output ket modes {m} overlap")
        if m := sets[3] & sets[7]:
            raise ValueError(f"input ket modes {m} overlap")
        bra_out = (
            sets[0] | sets[4]
        )  # (self.output.bra - other.input.bra) | other.output.bra
        bra_in = (
            sets[1] | sets[5]
        )  # self.input.bra | (other.input.bra - self.output.bra)
        ket_out = (
            sets[2] | sets[6]
        )  # (self.output.ket - other.input.ket) | other.output.ket
        ket_in = (
            sets[3] | sets[7]
        )  # self.input.ket | (other.input.ket - self.output.ket)
        w = Wires(bra_out, bra_in, ket_out, ket_in)

        # preserve ids
        for t in (0, 1, 2, 3):
            for m in w.args[t]:
                w.ids_dicts[t][m] = (
                    self.ids_dicts[t][m] if m in sets[t] else other.ids_dicts[t][m]
                )

        # calculate permutation
        result_ids = [id for d in w.ids_dicts for id in d.values()]
        self_other_ids = [
            self.ids_dicts[t][m] for t in (0, 1, 2, 3) for m in sorted(sets[t])
        ] + [other.ids_dicts[t][m] for t in (0, 1, 2, 3) for m in sorted(sets[t + 4])]
        perm = [self_other_ids.index(id) for id in result_ids]
        return w, perm

    def __repr__(self) -> str:
        return f"Wires{self.args}"

    def _repr_html_(self):  # pragma: no cover
        template = Template(filename=os.path.dirname(__file__) + "/assets/wires.txt")
        display(HTML(template.render(wires=self)))
