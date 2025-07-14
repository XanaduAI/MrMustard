"""Module containing wire classes for quantum and classical channels."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from random import randint

from IPython.display import display

from mrmustard import widgets

__all__ = ["Wires"]

"""
This module provides wire functionality for applications in MrMustard.
It defines the core classes for representing quantum and classical wires, and their
relationships in quantum optical circuits.
"""


class LegibleEnum(Enum):
    """Enum class that provides a more legible string representation."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class ReprEnum(LegibleEnum):
    """Enumeration of possible representations for quantum states and operations."""

    UNSPECIFIED = auto()
    BARGMANN = auto()
    FOCK = auto()
    QUADRATURE = auto()
    PHASESPACE = auto()
    CHARACTERISTIC = auto()


class WiresType(LegibleEnum):
    """Enumeration of possible wire types in quantum circuits."""

    DM_LIKE = auto()  # only output ket and bra on same modes
    KET_LIKE = auto()  # only output ket
    UNITARY_LIKE = auto()  # such that can map ket to ket
    CHANNEL_LIKE = auto()  # such that can map dm to dm
    PROJ_MEAS_LIKE = auto()  # only input ket
    POVM_LIKE = auto()  # only input ket and input bra on same modes
    CLASSICAL_LIKE = auto()  # only classical wires


@dataclass
class QuantumWire:
    """
    Represents a quantum wire in a circuit.

    Args:
        mode: The mode number this wire represents.
        is_out: Whether this is an output wire.
        is_ket: Whether this wire is on the ket side.
        index: The index of this wire in the circuit.
        repr: The representation of this wire.
        fock_cutoff: The (optional) Fock cutoff for this wire.
        id: Unique identifier for this wire.
    """

    mode: int
    is_out: bool
    is_ket: bool
    index: int
    repr: ReprEnum = ReprEnum.BARGMANN
    fock_cutoff: int | None = None
    id: int = field(default_factory=lambda: randint(0, 2**32 - 1), compare=False)

    def copy(self, new_id: bool = False) -> QuantumWire:
        """Create a copy of the quantum wire.

        Args:
            new_id (bool): If True, generates a new ID for the copy. Defaults to False.

        Returns:
            QuantumWire: A copy of the quantum wire
        """
        return QuantumWire(
            mode=self.mode,
            is_out=self.is_out,
            is_ket=self.is_ket,
            index=self.index,
            repr=self.repr,
            fock_cutoff=self.fock_cutoff,
            id=self.id if not new_id else randint(0, 2**32 - 1),
        )

    def _order(self) -> int:
        """
        Artificial ordering for sorting quantum wires.
        Order achieved is by bra/ket, then out/in, then mode.
        """
        return self.mode + 10_000 * (1 - 2 * self.is_out) - 100_000 * (1 - 2 * self.is_ket)

    def __eq__(self, other: QuantumWire) -> bool:
        return (
            self.mode == other.mode
            and self.is_out == other.is_out
            and self.is_ket == other.is_ket
            and self.repr == other.repr
        )

    def __hash__(self) -> int:
        return hash((self.mode, self.is_out, self.is_ket, self.repr))


@dataclass
class ClassicalWire:
    """
    Represents a classical wire in a circuit.

    Args:
        mode: The mode number this wire represents
        is_out: Whether this is an output wire
        index: The index of this wire in the circuit
        repr: The representation of this wire
        id: Unique identifier for this wire
    """

    mode: int
    is_out: bool
    index: int
    repr: ReprEnum = ReprEnum.UNSPECIFIED
    id: int = field(default_factory=lambda: randint(0, 2**32 - 1))

    def copy(self, new_id: bool = False) -> ClassicalWire:
        """Returns a copy of the classical wire."""
        return ClassicalWire(
            mode=self.mode,
            is_out=self.is_out,
            index=self.index,
            repr=self.repr,
            id=self.id if not new_id else randint(0, 2**32 - 1),
        )

    def _order(self) -> int:
        """
        Artificial ordering for sorting classical wires.
        Order is by out/in, then mode. Classical wires always come after quantum wires.
        """
        return 1000_000 + self.mode + 10_000 * (1 - 2 * self.is_out)

    def __eq__(self, other: ClassicalWire) -> bool:
        return self.mode == other.mode and self.is_out == other.is_out and self.repr == other.repr

    def __hash__(self) -> int:
        return hash((self.mode, self.is_out, self.repr))


class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, instances of ``CircuitComponent`` have a ``Wires`` attribute.
    The wires describe how they connect with the surrounding components in a tensor network picture,
    where states flow from left to right. ``CircuitComponent``\s can have wires on the
    bra and/or on the ket side. Additionally, they may have classical wires. Here are some examples
    for the types of components available on ``mrmustard.lab``:

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


       A ket representing the state of mode ``1`` has only output wires:

                        ╔═════════╗   1  ┌───────┐
                        ║   Ket   ║─────▶│Ket out│
                        ╚═════════╝      └───────┘

       A measurement acting on mode ``0`` has input wires on the ket side and classical output wires:

       ┌──────┐   0  ╔═════════════╗   0  ┌─────────────┐
       │Ket in│─────▶║ Measurement ║─────▶│Classical out│
       └──────┘      ╚═════════════╝      └─────────────┘

    The ``Wires`` class can then be used to create subsets of wires:

    .. code-block::

        >>> from mrmustard.physics.wires import Wires

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

                     ╔═════════════╗
        1 (2) ─────▶ ║             ║─────▶ 0 (0)
        2 (3) ─────▶ ║             ║─────▶ 1 (1)
                     ║             ║
                     ║  ``Wires``  ║
        1 (6) ─────▶ ║             ║
        2 (7) ─────▶ ║             ║─────▶ 0 (4)
       13 (8) ─────▶ ║             ║─────▶ 13 (5)
                     ╚═════════════╝

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

    The permutations that standardize the CV and DV variables of the contracted reprs are also returned.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
        classical_out: The output modes for classical information.
        classical_in: The input modes for classical information.

    Returns:
        A ``Wires`` object, and the permutations that standardize the CV and DV variables.
    """

    def __init__(
        self,
        modes_out_bra: set[int] | None = None,
        modes_in_bra: set[int] | None = None,
        modes_out_ket: set[int] | None = None,
        modes_in_ket: set[int] | None = None,
        classical_out: set[int] | None = None,
        classical_in: set[int] | None = None,
    ):
        modes_out_bra = modes_out_bra or set()
        modes_in_bra = modes_in_bra or set()
        modes_out_ket = modes_out_ket or set()
        modes_in_ket = modes_in_ket or set()
        classical_out = classical_out or set()
        classical_in = classical_in or set()

        self._quantum_wires = set()
        self._classical_wires = set()
        for i, m in enumerate(sorted(modes_out_bra)):
            self._quantum_wires.add(QuantumWire(mode=m, is_out=True, is_ket=False, index=i))
        n = len(modes_out_bra)
        for i, m in enumerate(sorted(modes_in_bra)):
            self._quantum_wires.add(QuantumWire(mode=m, is_out=False, is_ket=False, index=n + i))
        n += len(modes_in_bra)
        for i, m in enumerate(sorted(modes_out_ket)):
            self._quantum_wires.add(QuantumWire(mode=m, is_out=True, is_ket=True, index=n + i))
        n += len(modes_out_ket)
        for i, m in enumerate(sorted(modes_in_ket)):
            self._quantum_wires.add(QuantumWire(mode=m, is_out=False, is_ket=True, index=n + i))
        n += len(modes_in_ket)
        for i, m in enumerate(sorted(classical_out)):
            self._classical_wires.add(ClassicalWire(mode=m, is_out=True, index=n + i))
        n += len(classical_out)
        for i, m in enumerate(sorted(classical_in)):
            self._classical_wires.add(ClassicalWire(mode=m, is_out=False, index=n + i))

    @cached_property
    def adjoint(self) -> Wires:
        r"""
        New ``Wires`` object with the adjoint quantum wires (ket becomes bra and vice versa).
        """
        ret = self.copy(new_ids=True)
        for w in ret.quantum:
            w.is_ket = not w.is_ket
        ret._clear_cached_properties()
        ret._reindex()
        return ret

    @property
    def args(self) -> tuple[tuple[int, ...], ...]:
        r"""
        The arguments needed to create a new ``Wires`` object with the same wires.
        """
        return (
            self.bra.output.modes,
            self.bra.input.modes,
            self.ket.output.modes,
            self.ket.input.modes,
            self.classical.output.modes,
            self.classical.input.modes,
        )

    @cached_property
    def bra(self) -> Wires:
        r"""
        New ``Wires`` object with references to only quantum bra wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(quantum={q for q in self._quantum_wires if not q.is_ket})

    @cached_property
    def classical(self) -> Wires:
        r"""
        New ``Wires`` object with references to only classical wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(classical=self._classical_wires)

    @cached_property
    def dual(self) -> Wires:
        r"""
        New ``Wires`` object with dual quantum and classical wires (input becomes output and vice versa).
        """
        ret = self.copy(new_ids=True)
        for w in ret:
            w.is_out = not w.is_out
        ret._clear_cached_properties()
        ret._reindex()
        return ret

    @property
    def ids(self) -> tuple[int, ...]:
        r"""
        The ids of the wires in standard order.
        """
        return tuple(w.id for w in self.sorted_wires)

    @property
    def indices(self) -> tuple[int, ...]:
        r"""
        The indices of the wires in standard order.
        """
        return tuple(w.index for w in self.sorted_wires)

    @cached_property
    def input(self) -> Wires:
        r"""
        New ``Wires`` object with references to only classical and quantum input wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(
            quantum={q for q in self._quantum_wires if not q.is_out},
            classical={c for c in self._classical_wires if not c.is_out},
        )

    @cached_property
    def ket(self) -> Wires:
        r"""
        New ``Wires`` object with references to only quantum ket wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(quantum={q for q in self._quantum_wires if q.is_ket})

    @property
    def modes(self) -> set[int]:
        r"""
        The modes spanned by the wires.
        """
        return {q.mode for q in self._quantum_wires} | {c.mode for c in self._classical_wires}

    @cached_property
    def output(self) -> Wires:
        r"""
        New ``Wires`` object with references to only classical and quantum output wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(
            quantum={q for q in self._quantum_wires if q.is_out},
            classical={c for c in self._classical_wires if c.is_out},
        )

    @cached_property
    def quantum(self) -> Wires:
        r"""
        New ``Wires`` object with references to only quantum wires.
        Note that the wires are not copied.
        """
        return Wires.from_wires(quantum=self._quantum_wires)

    @cached_property
    def sorted_wires(self) -> list[QuantumWire | ClassicalWire]:
        r"""
        A list of all wires sorted in standard order.
        """
        return sorted({*self._quantum_wires, *self._classical_wires}, key=lambda s: s._order())

    @classmethod
    def from_wires(
        cls,
        quantum: Iterable[QuantumWire] = (),
        classical: Iterable[ClassicalWire] = (),
        copy: bool = False,
    ) -> Wires:
        r"""
        Returns a new Wires object with references to the given wires.
        If copy is True, the wires are copied, otherwise they are referenced.
        Does not reindex the wires.
        """
        w = cls()
        w._quantum_wires = set(quantum) if not copy else {q.copy() for q in quantum}
        w._classical_wires = set(classical) if not copy else {c.copy() for c in classical}
        return w

    def contracted_indices(self, other: Wires) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""
        Returns the indices (in standard order) being contracted between self and other when
        calling matmul.

        Args:
            other: another Wires object
        """
        ovlp_bra, ovlp_ket = self.overlap(other)
        idxA = self.output.bra[ovlp_bra].indices + self.output.ket[ovlp_ket].indices
        idxB = other.input.bra[ovlp_bra].indices + other.input.ket[ovlp_ket].indices
        return idxA, idxB

    def contracted_labels(self, other: Wires) -> tuple[list[int], list[int], list[int]]:
        r"""
        Returns the integer labels of the contracted wires, such that contracted wires have the same
        label. The output labels are sorted in standard order.

        Args:
            other: another Wires object
        """
        # Make a local copy of other with new ids to avoid conflicts
        other_copy = other.copy(new_ids=True)

        idxA, idxB = self.contracted_indices(other_copy)
        lblA = list(range(len(self)))
        lblB = list(range(len(self), len(self) + len(other_copy)))
        for i, j in zip(idxA, idxB):
            lblB[j] = lblA[i]
        output_labels = set(lblA) ^ set(lblB)
        id2label = {w.id: lbl for w, lbl in zip(self.sorted_wires, lblA)}
        id2label.update({w.id: lbl for w, lbl in zip(other_copy.sorted_wires, lblB)})
        wires_out, _ = self @ other_copy
        lbl_out = [
            id2label[w.id] for w in wires_out.sorted_wires if id2label[w.id] in output_labels
        ]
        return lblA, lblB, lbl_out

    def copy(self, new_ids: bool = False) -> Wires:
        """Returns a deep copy of this Wires object."""
        return Wires.from_wires(
            quantum={q.copy(new_ids) for q in self._quantum_wires},
            classical={c.copy(new_ids) for c in self._classical_wires},
        )

    def overlap(self, other: Wires) -> tuple[set[int], set[int]]:
        r"""
        Returns the modes that overlap between self and other.

        Args:
            other: Another ``Wires`` object.
        """
        ovlp_ket = self.output.ket.modes & other.input.ket.modes
        ovlp_bra = self.output.bra.modes & other.input.bra.modes
        return ovlp_bra, ovlp_ket

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        display(widgets.wires(self))

    def _clear_cached_properties(self) -> None:
        r"""
        Clears the cached properties of the Wires object.
        Note: This is required whenever the Wires object has been mutated to
        ensure it's properties are recomputed.
        """
        for name, value in inspect.getmembers(Wires):
            if isinstance(value, functools.cached_property):
                self.__dict__.pop(name, None)

    def _reindex(self) -> None:
        r"""
        Updates the indices of the wires according to the standard order.
        """
        for i, w in enumerate(self.sorted_wires):
            w.index = i

    def __add__(self, other: Wires) -> Wires:
        r"""
        New ``Wires`` object with references to the wires of self and other.
        If there are overlapping wires (same mode, is_ket, is_out), raises a ValueError.
        Note that the wires are not reindexed nor copied. Use with caution.
        """
        if ovlp_classical := self._classical_wires & other._classical_wires:
            raise ValueError(f"Overlapping classical wires {ovlp_classical}.")
        if ovlp_quantum := self._quantum_wires & other._quantum_wires:
            raise ValueError(f"Overlapping quantum wires {ovlp_quantum}.")
        return Wires.from_wires(
            quantum=self._quantum_wires | other._quantum_wires,
            classical=self._classical_wires | other._classical_wires,
        )

    def __bool__(self) -> bool:
        return bool(self._quantum_wires) or bool(self._classical_wires)

    def __eq__(self, other: Wires) -> bool:
        return self.args == other.args

    def __getitem__(self, modes: tuple[int, ...] | int) -> Wires:
        r"""
        Returns a new Wires object with references to the quantum and classical wires with the given modes.
        """
        modes = {modes} if isinstance(modes, int) else set(modes)
        return Wires.from_wires(
            quantum={q for q in self._quantum_wires if q.mode in modes},
            classical={c for c in self._classical_wires if c.mode in modes},
        )

    def __hash__(self) -> int:
        return hash((tuple(self._classical_wires), tuple(self._quantum_wires)))

    def __iter__(self) -> Iterator[QuantumWire | ClassicalWire]:
        return iter(self.sorted_wires)

    def __len__(self) -> int:
        return len(self._quantum_wires) + len(self._classical_wires)

    def __matmul__(self, other: Wires) -> tuple[Wires, list[int], list[int]]:
        r"""
        Returns the ``Wires`` for the circuit component resulting from the composition of self and other.
        Returns also the permutations of the CV and DV wires to reorder the wires to standard order.
        Consider the following example:

        .. code-block::

                ╔═══════╗           ╔═══════╗
            B───║ self  ║───A   D───║ other ║───C
            b───║       ║───a   d───║       ║───c
                ╚═══════╝           ╚═══════╝

        B and D-A must not overlap, same for b and d-a, etc. The result is a new ``Wires`` object

        .. code-block::

                       ╔═══════╗
            B+(D-A)────║self @ ║────C+(A-D)
            b+(d-a)────║ other ║────c+(a-d)
                       ╚═══════╝

        Using the permutations, it is possible to write:

        .. code-block::

            ansatz = ansatz1[idx1] @ ansatz2[idx2]  # not in standard order
            wires, perm_CV, perm_DV = wires1 @ wires2  # matmul the wires
            ansatz = ansatz.reorder(perm_CV, perm_DV)  # now in standard order

        Args:
            other: The wires of the other circuit component.

        Returns:
            The wires of the circuit composition and the permutations.
        """
        bra_out = other.output.bra + (self.output.bra - other.input.bra)
        ket_out = other.output.ket + (self.output.ket - other.input.ket)
        bra_in = self.input.bra + (other.input.bra - self.output.bra)
        ket_in = self.input.ket + (other.input.ket - self.output.ket)
        cl_out = other.classical.output + (self.classical.output - other.classical.input)
        cl_in = self.classical.input + (other.classical.input - self.classical.output)

        # get the wires
        new_wires = Wires.from_wires(
            quantum=bra_out.sorted_wires
            + bra_in.sorted_wires
            + ket_out.sorted_wires
            + ket_in.sorted_wires,
            classical=cl_out.sorted_wires + cl_in.sorted_wires,
            copy=True,  # because we will call _reindex()
        )
        new_wires._reindex()

        combined = [w for w in self.sorted_wires if w.id in new_wires.ids] + [
            w for w in other.sorted_wires if w.id in new_wires.ids
        ]  # NOTE: assumes self and other have different ids
        perm = [combined.index(w) for w in new_wires.sorted_wires]
        return new_wires, perm

    def __repr__(self) -> str:
        return (
            f"Wires(modes_out_bra={self.output.bra.modes}, "
            f"modes_in_bra={self.input.bra.modes}, "
            f"modes_out_ket={self.output.ket.modes}, "
            f"modes_in_ket={self.input.ket.modes}, "
            f"classical_out={self.output.classical.modes}, "
            f"classical_in={self.input.classical.modes})"
        )

    def __sub__(self, other: Wires) -> Wires:
        r"""
        New ``Wires`` object with references to the wires of self whose modes are not in other.
        Note that the wires are not reindexed nor copied. Use with caution.
        """
        return Wires.from_wires(
            quantum={q for q in self._quantum_wires if q.mode not in other.modes},
            classical={c for c in self._classical_wires if c.mode not in other.modes},
        )
