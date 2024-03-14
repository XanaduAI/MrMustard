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
from functools import cached_property
from typing import Optional
import numpy as np

__all__ = ["Wires"]


class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, instances of ``CircuitComponent`` have a ``Wires`` attribute.
    The wires describe how they connect with the surrounding components in a circuit picture.
    For example, an `N`-mode pure state has `N` ket wires on the output side,
    while a `N`-mode mixed state has `N` ket wires and `N` bra wires on the output side.

    The ``Wires`` class is used to return the modes of the wires in a standard order:
    mode first, then bra/ket, then output/input. Note that although internally they are
    stored as sets of modes for performance reasons, the modes are to be considered ordered.

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

    To access the index of a subset of wires in standard order
    (i.e. skipping over wires not belonging to the subset),
    one can use the ``indices`` attribute:

    .. code-block::

        >>> w = Wires(modes_in_ket = {0,1}, modes_out_ket = {0,1})
        >>> assert w.indices == (0,1,2,3)
        >>> assert w.input.indices == (2,3)

    Note that subsets return new ``Wires`` objects.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra  (set[int]): The input modes on the bra side.
        modes_out_ket (set[int]): The output modes on the ket side.
        modes_in_ket  (set[int]): The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: Optional[set[int]] = None,
        modes_in_bra: Optional[set[int]] = None,
        modes_out_ket: Optional[set[int]] = None,
        modes_in_ket: Optional[set[int]] = None,
        original: Optional[Wires] = None,
        ids: Optional[list[int]] = None,
    ) -> None:

        self._mode_cache = {}
        self.args: tuple[set,...] = (
            modes_out_bra or set(),
            modes_in_bra or set(),
            modes_out_ket or set(),
            modes_in_ket or set(),
        )
        self._original = original
        # self._ids: list[int] | None = ids
    
    @cached_property
    def id(self) -> int:
        if self._original:
            return self._original.id
        return np.random.randint(0, 2**32)

    @cached_property
    def index_dicts(self) -> list[dict[int,int]]:
        r"The index dictionaries for the standard order. Only makes sense for original Wires."
        if self._original:
            return self._original.index_dicts
        return [{m:i + sum(len(s) for s in self.args[:t]) for i,m in enumerate(lst)} for t,lst in enumerate(self.sorted_args)]

    @cached_property
    def ids_dicts(self) -> list[dict[int,int]]:
        r"The id dictionaries for the standard order. Only makes sense for original Wires."
        if self._original:
            return self._original.ids_dicts
        return [{m:i + self.id for m,i in d.items()} for d in self.index_dicts]
    
    @cached_property
    def ids(self) -> list[int]:
        r"The ids of the indices standard order with sorted modes."
        return list(i + self.id for i in self.indices)    

    @cached_property
    def sorted_args(self) -> tuple[list[int], ...]:
        r"The sorted arguments."
        return tuple(sorted(s) for s in self.args)

    # @property
    # def ids(self) -> tuple[int, ...]:
    #     r"The ids of the indices standard order with sorted modes."
    #     if not self._ids:
    #         self._ids = tuple(i + self.original.id for i in self.indices)
    #     return self._ids

    @cached_property
    def original(self):
        r"The 'parent' ``Wires`` object, if any."
        return self._original or self

    # @cached_property
    # def types(self) -> set[int]:
    #     r"A set of up to four integers representing the types of wires in the standard order."
    #     return set(i for i in (0, 1, 2, 3) if bool(self.args[i]))

    @cached_property
    def modes(self) -> set[int]:
        r"The modes spanned by the wires."
        return set.union(*self.args)

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
        return tuple(self.original.index_dicts[t][m] for t,modes in enumerate(self.sorted_args) for m in modes)

    @cached_property
    def input(self) -> Wires:
        r"A view of this ``Wires`` object without output wires."
        return Wires(set(), self.args[1], set(), self.args[3], self.original)

    @cached_property
    def output(self) -> Wires:
        r"A view of this ``Wires`` object without input wires."
        return Wires(self.args[0], set(), self.args[2], set(), self.original)

    @cached_property
    def ket(self) -> Wires:
        r"A view of this ``Wires`` object without bra wires."
        return Wires(set(), set(), self.args[2], self.args[3], self.original)

    @cached_property
    def bra(self) -> Wires:
        r"A view of this ``Wires`` object without ket wires."
        return Wires(self.args[0], self.args[1], set(), set(), self.original)

    @cached_property
    def adjoint(self) -> Wires:
        r"A new ``Wires`` object obtained by swapping ket and bra wires."
        return Wires(self.args[2], self.args[3], self.args[0], self.args[1])

    @cached_property
    def dual(self) -> Wires:
        r"A new ``Wires`` object obtained by swapping input and output wires."
        return Wires(self.args[1], self.args[0], self.args[3], self.args[2])

    def __hash__(self) -> int:  # for getitem caching
        return hash(tuple(s) for s in self.args)

    def __getitem__(self, modes: tuple[int, ...] | int) -> Wires:
        r"A view of this Wires object with wires only on the given modes."
        modes_set = {modes} if isinstance(modes, int) else set(modes)
        if modes not in self._mode_cache:
            self._mode_cache[modes] = Wires(
                *(self.args[t] & modes_set for t in (0, 1, 2, 3)),
                original=self.original,
            )
        return self._mode_cache[modes]

    def __add__(self, other: Wires) -> Wires:
        r"""
        A new ``Wires`` object that combines the wires of ``self`` and those of ``other``.
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
        return len(self.modes) > 0

    def __eq__(self, other) -> bool:
        return self.args == other.args

    # pylint: disable=too-many-branches
    def __matmul__(self, other: Wires) -> tuple[Wires, list[int]]:
        r"""
        Returns the wires of the circuit composition of self and other without adding missing
        adjoints. It also returns the permutation that takes the contracted representations
        to the standard order. An exception is raised if any leftover wires would overlap.

        In pseudocode, we have:
        ``B[self]A  @  D[other]C  =  sort(B|(D-A))[result]sort(C|(A-D))``. The standard order
        would then be ``sort(C|(A-D))+sort(B|(D-A))`` (out, then in).
        In comparison, contracting the representations rather than the wires corresponds to
        an order where we start from juxtaposing the objects and then removing pairs of contracted
        indices, from self.output and other.input.
        ``sorted(A-D)+sorted(B)+sorted(C)+sorted(D-A)``, where each of the four parts is offset
        by the length of the previous ones. The returned permutation is the one that takes the
        result of multiplying representations to the standard order.

        Args:
            other (Wires): The wires of the other circuit component.
        Returns:
            tuple[Wires, list[int]]: The wires of the circuit composition and the permutation.
        Raises:
            ValueError: If any leftover wires would overlap.
        """
        if self._original or other._original:
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
        bra_out = sets[0] | sets[4]  # A-D | C i.e. (self.output.bra - other.input.bra) | other.output.bra
        bra_in  = sets[1] | sets[5]  # B | D-A i.e. self.input.bra | (other.input.bra - self.output.bra)
        ket_out = sets[2] | sets[6]  # a-d | c i.e. (self.output.ket - other.input.ket) | other.output.ket
        ket_in  = sets[3] | sets[7]  # b | d-a i.e. self.input.ket | (other.input.ket - self.output.ket)
        w = Wires(bra_out, bra_in, ket_out, ket_in)
        for t in (0,1,2,3):
            for m in w.args[t]:
                w.ids_dicts[t][m] = self.original.ids_dicts[t][m] if m in sets[t] else other.original.ids_dicts[t][m]
        # calculate permutation
        repr_index = ([ self.original.index_dicts[t][m] for t,s in enumerate(sets[:4]) for m in s] +
                      [other.original.index_dicts[t][m] for t,s in enumerate(sets[4:]) for m in s])
        perm = [w.indices.index(i) for i in repr_index]
        return w, perm

    def __repr__(self) -> str:
        return f"Wires{self.args}"
