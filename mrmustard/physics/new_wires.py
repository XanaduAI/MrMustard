from __future__ import annotations
from dataclasses import dataclass, field
from random import randint
from copy import deepcopy
from functools import cached_property, lru_cache
from enum import Enum, auto
from IPython.display import display

from mrmustard import widgets

__all__ = ["Wires"]


class Repr(Enum):
    UNSPECIFIED = auto()
    BARGMANN = auto()
    FOCK = auto()
    QUADRATURE = auto()
    PHASESPACE = auto()
    CHARACTERISTIC = auto()


@dataclass
class QuantumWire:
    mode: int
    is_out: bool
    is_ket: bool
    index: int
    repr: Repr = Repr.UNSPECIFIED
    id: int = field(default_factory=lambda: randint(0, 2**32 - 1))

    @property
    def is_dv(self) -> bool:
        return self.repr == Repr.FOCK

    def __hash__(self) -> int:
        return hash((self.mode, self.is_out, self.is_ket))

    def __repr__(self) -> str:
        return f"QuantumWire(mode={self.mode}, out={self.is_out}, ket={self.is_ket}, dv={self.is_dv}, repr={self.repr}, index={self.index})"

    def __eq__(self, other: QuantumWire) -> bool:
        return (
            self.mode == other.mode
            and self.is_out == other.is_out
            and self.is_ket == other.is_ket
            and self.is_dv == other.is_dv
            and self.repr == other.repr
        )


@dataclass
class ClassicalWire:
    mode: int
    is_out: bool
    index: int
    repr: Repr = Repr.UNSPECIFIED
    id: int = field(default_factory=lambda: randint(0, 2**32 - 1))

    @property
    def is_dv(self) -> bool:
        return self.repr == Repr.FOCK

    def __hash__(self) -> int:
        return hash((self.mode, self.is_out, self.is_dv))

    def __repr__(self) -> str:
        return f"ClassicalWire(mode={self.mode}, out={self.is_out}, dv={self.is_dv}, index={self.index})"

    def __eq__(self, other: ClassicalWire) -> bool:
        return self.mode == other.mode and self.is_out == other.is_out and self.is_dv == other.is_dv


class Wires:
    r"""
    A class with wire functionality for tensor network applications.

    In MrMustard, instances of ``CircuitComponent`` have a ``Wires`` attribute.
    The wires describe how they connect with the surrounding components in a tensor network picture,
    where states flow from left to right. ``CircuitComponent``\s can have wires on the
    bra and/or on the ket side. Additionally, they may have classical wires. Here are some examples
    for the types of components available on ``mrmustard.lab_dev``:

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

    The permutations that standardize the CV and DV variables of the contracted representations are also returned.

    Args:
        modes_out_bra: The output modes on the bra side.
        modes_in_bra: The input modes on the bra side.
        modes_out_ket: The output modes on the ket side.
        modes_in_ket: The input modes on the ket side.
        classical_out: The output modes for classical information.
        classical_in: The input modes for classical information.
        FOCK: The modes that are in Fock representation.

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
        FOCK: set[int] | None = None,
    ):
        self.id = randint(0, 2**32 - 1)
        self.quantum_wires = set()
        self.classical_wires = set()
        self.FOCK = FOCK or set()

        for i, m in enumerate(sorted(modes_out_bra or [])):
            self.quantum_wires.add(
                QuantumWire(
                    mode=m,
                    is_out=True,
                    is_ket=False,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=i,
                )
            )
        n = len(modes_out_bra or [])
        for i, m in enumerate(sorted(modes_in_bra or [])):
            self.quantum_wires.add(
                QuantumWire(
                    mode=m,
                    is_out=False,
                    is_ket=False,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=n + i,
                )
            )
        n += len(modes_in_bra or [])
        for i, m in enumerate(sorted(modes_out_ket or [])):
            self.quantum_wires.add(
                QuantumWire(
                    mode=m,
                    is_out=True,
                    is_ket=True,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=n + i,
                )
            )
        n += len(modes_out_ket or [])
        for i, m in enumerate(sorted(modes_in_ket or [])):
            self.quantum_wires.add(
                QuantumWire(
                    mode=m,
                    is_out=False,
                    is_ket=True,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=n + i,
                )
            )
        n += len(modes_in_ket or [])
        for i, m in enumerate(sorted(classical_out or [])):
            self.classical_wires.add(
                ClassicalWire(
                    mode=m,
                    is_out=True,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=n + i,
                )
            )
        n += len(classical_out or [])
        for i, m in enumerate(sorted(classical_in or [])):
            self.classical_wires.add(
                ClassicalWire(
                    mode=m,
                    is_out=False,
                    repr=Repr.FOCK if m in self.FOCK else Repr.UNSPECIFIED,
                    index=n + i,
                )
            )

    def copy(self) -> Wires:
        return deepcopy(self)

    ###### TRANSFORMATIONS ######

    @cached_property
    def adjoint(self) -> Wires:
        r"""
        New ``Wires`` object with the adjoint quantum wires (ket becomes bra and vice versa).
        """
        w = self.copy()
        for q in w.quantum_wires:
            q.is_ket = not q.is_ket
        return w

    @cached_property
    def dual(self) -> Wires:
        r"""
        New ``Wires`` object with dual quantum and classical wires (input becomes output and vice versa).
        """
        w = self.copy()
        for q in w.quantum_wires:
            q.is_out = not q.is_out
        for c in w.classical_wires:
            c.is_out = not c.is_out
        return w

    ###### SUBSETS ######

    @lru_cache
    def __getitem__(self, modes: tuple[int, ...] | int) -> Wires:
        """
        Returns the quantum and classical wires with the given modes.
        """
        modes = {modes} if isinstance(modes, int) else set(modes)
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if q.mode in modes}
        w.classical_wires = {c for c in self.classical_wires.copy() if c.mode in modes}
        return w

    @cached_property
    def classical(self) -> Wires:
        r"""
        New ``Wires`` object with only classical wires.
        """
        w = Wires()
        w.classical_wires = self.classical_wires.copy()
        return w

    @cached_property
    def quantum(self) -> Wires:
        r"""
        New ``Wires`` object with only quantum wires.
        """
        w = Wires()
        w.quantum_wires = self.quantum_wires.copy()
        return w

    @cached_property
    def bra(self) -> Wires:
        r"""
        New ``Wires`` object with only quantum bra wires.
        """
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if not q.is_ket}
        return w

    @cached_property
    def ket(self) -> Wires:
        r"""
        New ``Wires`` object with only quantum ket wires.
        """
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if q.is_ket}
        return w

    @cached_property
    def input(self) -> Wires:
        r"""
        New ``Wires`` object with only classical and quantum input wires.
        """
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if not q.is_out}
        w.classical_wires = {c for c in self.classical_wires.copy() if not c.is_out}
        return w

    @cached_property
    def output(self) -> Wires:
        r"""
        New ``Wires`` object with only classical and quantum output wires.
        """
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if q.is_out}
        w.classical_wires = {c for c in self.classical_wires.copy() if c.is_out}
        return w

    ###### PROPERTIES ######

    @cached_property
    def ids(self) -> tuple[int, ...]:
        r"""
        The ids of the wires in standard order.
        """
        return tuple(w.id for w in self.sorted_wires)

    @cached_property
    def DV_indices(self) -> tuple[int, ...]:
        r"""
        The indices of the DV wires (both quantum and classical) in standard order.
        """
        return tuple(q.index for q in self.DV_wires)

    @cached_property
    def CV_indices(self) -> tuple[int, ...]:
        r"""
        The indices of the CV wires (both quantum and classical) in standard order.
        """
        return tuple(q.index for q in self.CV_wires)

    @cached_property
    def DV_wires(self) -> tuple[QuantumWire | ClassicalWire, ...]:
        r"""
        The DV wires in standard order.
        """
        return tuple(w for w in self.sorted_wires if w.is_dv)

    @cached_property
    def indices(self) -> tuple[int, ...]:
        r"""
        The indices of the wires in standard order.
        """
        return tuple(w.index for w in self.sorted_wires)

    @cached_property
    def CV_wires(self) -> tuple[QuantumWire | ClassicalWire, ...]:
        r"""
        The CV wires in standard order.
        """
        return tuple(w for w in self.sorted_wires if not w.is_dv)

    @cached_property
    def modes(self) -> set[int]:
        r"""
        The modes spanned by the wires.
        """
        return {q.mode for q in self.quantum_wires} | {c.mode for c in self.classical_wires}

    @cached_property
    def args(self) -> tuple[set[int], ...]:
        r"""
        The arguments to pass to ``Wires`` to create the same object.
        """
        return (
            self.bra.output.modes,
            self.bra.input.modes,
            self.ket.output.modes,
            self.ket.input.modes,
            self.classical.output.modes,
            self.classical.input.modes,
            self.FOCK,
        )

    @cached_property
    def wires(self) -> set[QuantumWire | ClassicalWire]:
        r"""
        A set of all wires.
        """
        return {*self.quantum_wires, *self.classical_wires}

    @cached_property
    def sorted_wires(self) -> list[QuantumWire | ClassicalWire]:
        r"""
        A list of all wires in standard order.
        """
        return [
            *sorted(self.bra.output.wires, key=lambda s: s.mode),
            *sorted(self.bra.input.wires, key=lambda s: s.mode),
            *sorted(self.ket.output.wires, key=lambda s: s.mode),
            *sorted(self.ket.input.wires, key=lambda s: s.mode),
            *sorted(self.classical.output.wires, key=lambda s: s.mode),
            *sorted(self.classical.input.wires, key=lambda s: s.mode),
        ]

    ###### METHODS ######

    def reindex(self) -> None:
        r"""
        Updates the indices of the wires according to the standard order.
        """
        for i, w in enumerate(self.sorted_wires):
            w.index = i

    def __add__(self, other: Wires) -> Wires:
        r"""
        New ``Wires`` object that combines the wires of self and other.
        If there are overlapping wires (same mode, is_ket, is_out), raises a ValueError.
        """
        if ovlp_classical := self.classical_wires & other.classical_wires:
            raise ValueError(f"Overlapping classical wires {ovlp_classical}.")
        if ovlp_quantum := self.quantum_wires & other.quantum_wires:
            raise ValueError(f"Overlapping quantum wires {ovlp_quantum}.")
        w = Wires()
        w.quantum_wires = self.quantum_wires | other.quantum_wires
        w.classical_wires = self.classical_wires | other.classical_wires
        w.reindex()
        return w

    def __sub__(self, other: Wires) -> Wires:
        r"""
        New ``Wires`` object that removes the wires of other from self, by mode.
        Note it does not look at ket, bra, input or output: just the mode. Use with caution.
        """
        w = Wires()
        w.quantum_wires = {q for q in self.quantum_wires.copy() if q.mode not in other.modes}
        w.classical_wires = {c for c in self.classical_wires.copy() if c.mode not in other.modes}
        w.reindex()
        return w

    def __bool__(self) -> bool:
        return bool(self.quantum_wires) or bool(self.classical_wires)

    def __hash__(self) -> int:
        return hash(tuple(tuple(sorted(subset)) for subset in self.args))

    def __eq__(self, other: Wires) -> bool:
        return (
            self.quantum_wires == other.quantum_wires
            and self.classical_wires == other.classical_wires
        )

    def __len__(self) -> int:
        return len(self.quantum_wires) + len(self.classical_wires)

    def __repr__(self) -> str:
        return (
            f"Wires(modes_out_bra={self.output.bra.modes}, "
            f"modes_in_bra={self.input.bra.modes}, "
            f"modes_out_ket={self.output.ket.modes}, "
            f"modes_in_ket={self.input.ket.modes}, "
            f"classical_out={self.output.classical.modes}, "
            f"classical_in={self.input.classical.modes}, "
            f"FOCK={self.FOCK})"
        )

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
        w = Wires()
        w.quantum_wires = (bra_out + bra_in + ket_out + ket_in).wires
        w.classical_wires = (cl_out + cl_in).wires
        w.reindex()

        # get the permutations
        CV_ids = [w.id for w in w.CV_wires if w.id in self.ids] + [
            w.id for w in w.CV_wires if w.id in other.ids
        ]
        DV_ids = [w.id for w in w.DV_wires if w.id in self.ids] + [
            w.id for w in w.DV_wires if w.id in other.ids
        ]
        CV_perm = [CV_ids.index(w.id) for w in w.CV_wires]
        DV_perm = [DV_ids.index(w.id) for w in w.DV_wires]
        return w, CV_perm, DV_perm

    def _ipython_display_(self):
        display(widgets.wires(self))
