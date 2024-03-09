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

# cached_property = property


class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, instances of `CircuitComponent` have a ``Wires`` attribute.
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
        >>> assert w.bra.modes == {1, 2}

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
        modes_out_bra (set[int]): The output modes on the bra side.
        modes_in_bra  (set[int]): The input modes on the bra side.
        modes_out_ket (set[int]): The output modes on the ket side.
        modes_in_ket  (set[int]): The input modes on the ket side.
    """

    def __init__(
        self,
        modes_out_bra: set[int] = set(),
        modes_in_bra: set[int] = set(),
        modes_out_ket: set[int] = set(),
        modes_in_ket: set[int] = set(),
        original: Optional[Wires] = None,
    ) -> None:

        self.mode_cache = {}
        self.args = modes_out_bra, modes_in_bra, modes_out_ket, modes_in_ket
        self._original = original

    @property
    def original(self):
        if self._original is None:
            return self
        return self._original

    @cached_property
    def types(self) -> set[int]:
        r"A set of up to four integers representing the types of wires in the standard order."
        return set([i for i in (0, 1, 2, 3) if bool(self.args[i])])

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
        a, b, c, _ = self.original.args
        d = (0, len(a), len(a) + len(b), len(a) + len(b) + len(c))
        return tuple(
            sorted(self.original.args[i]).index(m) + d[i]
            for i in (0, 1, 2, 3)
            for m in sorted(self.args[i])
        )

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
        if modes not in self.mode_cache:
            self.mode_cache[modes] = Wires(
                *(self.args[i] & modes_set for i in (0, 1, 2, 3)),
                original=Wires(*self.original.args),
            )
        return self.mode_cache[modes]

    def __add__(self, other: Wires) -> Wires:
        r"""
        A new ``Wires`` object that combines the wires of ``self`` and those of ``other``.
        Raises:
            ValueError: If any leftover wires would overlap.
        """
        new_args = []
        for t, (m1, m2) in enumerate(zip(self.args, other.args)):
            if m := (m1 & m2):
                raise ValueError(f"{t}-type wires overlap at mode {m}")
            new_args.append(m1 | m2)
        return Wires(*new_args)

    def __bool__(self) -> bool:
        r"Returns ``True`` if this ``Wires`` object has any wires, ``False`` otherwise."
        return len(self.modes) > 0

    def __eq__(self, other) -> bool:
        return self.args == other.args

    def __matmul__(self, other: Wires) -> tuple[Wires, tuple[int, ...]]:
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
        A, B, a, b = self.args
        C, D, c, d = other.args
        if m := C & (A - D):
            raise ValueError(f"output bra modes {m} overlap")
        if m := B & (D - A):
            raise ValueError(f"input bra modes {m} overlap")
        if m := c & (a - d):
            raise ValueError(f"output ket modes {m} overlap")
        if m := b & (d - a):
            raise ValueError(f"input ket modes {m} overlap")
        bra_out = C | (A - D)
        bra_in = B | (D - A)
        ket_out = c | (a - d)
        ket_in = b | (d - a)
        w = Wires(bra_out, bra_in, ket_out, ket_in)
        # calculate permutation from the contracted representation to the standard order
        sets = (A - D, B, a - d, b, C, D - A, c, d - a)
        lists = list(map(sorted, sets))
        offsets = list(map(len, sets))
        final = [sorted(bra_out), sorted(bra_in), sorted(ket_out), sorted(ket_in)]
        perm = []
        for m in final[0]:
            if m in sets[0]:
                perm.append(lists[0].index(m))
            else:
                perm.append(lists[4].index(m) + sum(offsets[:4]))
        for m in final[1]:
            if m in sets[1]:
                perm.append(lists[1].index(m) + sum(offsets[:1]))
            else:
                perm.append(lists[5].index(m) + sum(offsets[:5]))
        for m in final[2]:
            if m in sets[2]:
                perm.append(lists[2].index(m) + sum(offsets[:2]))
            else:
                perm.append(lists[6].index(m) + sum(offsets[:6]))
        for m in final[3]:
            if m in sets[3]:
                perm.append(lists[3].index(m) + sum(offsets[:3]))
            else:
                perm.append(lists[7].index(m) + sum(offsets[:7]))
        return w, tuple(perm)

    def __repr__(self) -> str:
        return f"Wires{self.args}"
