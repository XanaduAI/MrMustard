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
from functools import cached_property, cache

__all__ = ["Wires"]

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
        modes_out_bra: set[int] = set(),  # TODO switch to frozensets?
        modes_in_bra: set[int] = set(),
        modes_out_ket: set[int] = set(),
        modes_in_ket: set[int] = set(),
    ) -> None:

        self.ob = modes_out_bra
        self.ib = modes_in_bra
        self.ok = modes_out_ket
        self.ik = modes_in_ket
        self._exclude_modes: set[int] = set()
        self._exclude_types: set[int] = set()

    def view(self, exclude_types: set[int] = set(), exclude_modes: set[int] = set()) -> Wires:
        r"""
        A view of this Wires object with partially excluded types and modes.
        Used to model .ket, .bra, .input, .output and [modes] properties.

        Args:
            exclude_types (tuple[int,...]): The types of wires to exclude.
            exclude_modes (tuple[int,...]): The modes of wires to exclude.
        """
        w = Wires(*self.args)
        w._exclude_types = exclude_types
        w._exclude_modes = exclude_modes
        return w
    
    @cached_property
    def types(self) -> set[int]:
        r"A set of up to four integers representing the types of wires in the standard order."
        return set((0,) * bool(self.ob) + (1,) * bool(self.ib) + (2,) * bool(self.ok) + (3,) * bool(self.ik)) - self._exclude_types

    @cached_property
    def args(self) -> tuple[set[int],...]:
        r"Returns the input arguments needed to initialize the same ``Wires`` object."
        return (self.ob - self._exclude_modes if 0 in self.types else set(),
                self.ib - self._exclude_modes if 1 in self.types else set(),
                self.ok - self._exclude_modes if 2 in self.types else set(),
                self.ik - self._exclude_modes if 3 in self.types else set())

    @cached_property
    def modes(self) -> set[int]:
        r"The modes of the wires in the standard order."
        return set.union(*self.args)

    @cached_property
    def indices(self) -> tuple[int,...]:
        r"""
        The array of indices of this ``Wires`` in the standard order. The array of indices
        of this ``Wires`` in the standard order. When a subset is selected (e.g. ``.ket``),
        it skips the indices of wires that do not belong to the subset.

        .. code-block::

            >>> w = Wires(modes_in_ket = (0,1), modes_out_ket = (0,1))
            >>> assert w.indices == (0,1,2,3)
            >>> assert w.input.indices == (2,3)
        """
        modes = (sorted(self.ob), sorted(self.ib), sorted(self.ok), sorted(self.ik))
        d = (0, len(self.ob), len(self.ob) + len(self.ib), len(self.ob) + len(self.ib) + len(self.ok))
        return tuple(modes[t].index(m) + d[t] for t in (0,1,2,3) for m in sorted(self.modes & set(modes[t])))

    @cached_property
    def input(self) -> Wires:
        r"A view of this ``Wires`` object without output wires."
        return self.view(exclude_types = self._exclude_types | {0,2})

    @cached_property
    def output(self) -> Wires:
        r"A view of this ``Wires`` object without input wires."
        return self.view(exclude_types = self._exclude_types | {1,3})

    @cached_property
    def ket(self) -> Wires:
        r"A view of this ``Wires`` object without bra wires."
        return self.view(exclude_types = self._exclude_types | {0,1})
    
    @cached_property
    def bra(self) -> Wires:
        r"A view of this ``Wires`` object without ket wires."
        return self.view(exclude_types = self._exclude_types | {2,3})
    
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

    @cache
    def __getitem__(self, modes: tuple[int,...] | int) -> Wires:
        r"A view of this Wires object with wires only on the given modes."
        modes = (modes,) if isinstance(modes, int) else modes
        return self.view(exclude_modes = self.modes - set(modes))

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
        return len(self.indices) > 0

    def __eq__(self, other) -> bool:
        return self.args == other.args

    def __matmul__(self, other: Wires) -> tuple[Wires, list[int]]:
        r"""
        Returns the wires of the circuit composition of self and other (without adding missing adjoints)
        and the permutation that takes the contracted representations to the standard order.
        An exception is raised if any leftover wires would overlap.

        Indicating the input and output sets of modes of self and other in pseudocode as A,B,C,D, we have:
        ``B[self]A  @  D[other]C  =  sort(B+(D-A))[result]sort(C+(A-D))``. The standard order would then be:
        ``sort(C|(A-D))+sort(B|(D-A))``.
        In comparison, contracting the representations rather than the wires corresponds to an order:
        ``list(A-D)+list(B)+list(C)+list(D-A)``.
        The returned permutation is the one that takes the representation to the standard order.

        Args:
            other (Wires): The wires of the other circuit component.
        Returns:
            tuple[Wires, list[int]]: The wires of the circuit composition and the permutation.
        Raises:
            ValueError: If any leftover wires would overlap.
        """
        A, B, a, b = self.args
        C, D, c, d = other.args
        if (m := C & (A - D)):
            raise ValueError(f"output bra wires {m} overlap")
        if (m := B & (D - A)):
            raise ValueError(f"input bra wires {m} overlap")
        if (m := c & (a - d)):
            raise ValueError(f"output ket wires {m} overlap")
        if (m := b & (d - a)):
            raise ValueError(f"input ket wires {m} overlap")
        bra_out = sorted(C | (A - D))
        bra_in  = sorted(B | (D - A))
        ket_out = sorted(c | (a - d))
        ket_in  = sorted(b | (d - a))
        repr_order = list(A-D) + list(B) + list(a-d) + list(b) + list(C) + list(D-A) + list(c) + list(d-a)
        wires_order = bra_out + bra_in + ket_out + ket_in
        perm = [wires_order.index(m) for m in repr_order]
        return Wires(set(bra_out), set(bra_in), set(ket_out), set(ket_in)), perm

    def __repr__(self) -> str:
        return f"Wires{self.args}"
    