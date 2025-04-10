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

"""
This module contains the defintion of the ket class ``Ket``.
"""

from __future__ import annotations

from typing import Collection, Sequence
import warnings
from IPython.display import display

from mrmustard import math, settings, widgets
from mrmustard.math.lattice.strategies.vanilla import autoshape_numba
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_psi
from mrmustard.physics.gaussian import purity
from mrmustard.physics.utils import generate_batch_str
from mrmustard.physics.representations import Representation
from mrmustard.physics.wires import Wires, ReprEnum
from mrmustard.utils.typing import (
    Scalar,
    Batch,
)

from .base import State, _validate_operator, OperatorType
from .dm import DM
from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..utils import shape_check

__all__ = ["Ket"]


class Ket(State):
    r"""
    Base class for all Hilbert space vectors.
    """

    short_name = "Ket"

    @property
    def is_physical(self) -> bool:  # TODO: revisit
        r"""
        Whether the ket object is a physical one.
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError(
                "Physicality conditions are not implemented for a linear superposition of states."
            )
        if self.ansatz.num_derived_vars > 0:
            raise NotImplementedError(
                "Physicality conditions are not implemented for derived variables."
            )
        if isinstance(self.ansatz, ArrayAnsatz):
            raise NotImplementedError(
                "Physicality conditions are not implemented for states with ArrayAnsatz."
            )
        return math.allclose(math.abs(math.eigvals(self.ansatz.A)) < 1, True) and math.allclose(
            self.probability, 1, settings.ATOL
        )

    @property
    def probability(self) -> float:
        r"""Probability of this Ket (L2 norm squared)."""
        return self.L2_norm

    @property
    def purity(self) -> float:
        if self.ansatz:
            shape = (
                self.ansatz.batch_shape[:-1] if self.ansatz._lin_sup else self.ansatz.batch_shape
            )
        else:
            shape = ()
        return math.ones(shape)

    @classmethod
    def from_ansatz(
        cls,
        modes: Collection[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> State:
        if not isinstance(modes, set) and sorted(modes) != list(modes):
            raise ValueError(f"Modes must be sorted. Got {modes}")
        modes = set(modes)
        if ansatz and ansatz.num_vars != len(modes):
            raise ValueError(
                f"Expected an ansatz with {len(modes)} variables, found {ansatz.num_vars}."
            )
        wires = Wires(modes_out_ket=modes)
        if isinstance(ansatz, ArrayAnsatz):
            for w in wires.quantum_wires:
                w.repr = ReprEnum.FOCK
        return Ket(Representation(ansatz, wires), name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple,
        name: str | None = None,
        atol_purity: float | None = 1e-5,
    ) -> Ket:  # TODO: revisit
        cov, means, coeff = triple
        cov = math.astensor(cov)
        means = math.astensor(means)
        if cov.shape[:-2] != ():
            raise NotImplementedError("Not implemented for batched states.")
        shape_check(cov, means, 2 * len(modes), "Phase space")
        if atol_purity:
            p = float(purity(cov))
            math.error_if(
                p,
                p < 1.0 - atol_purity,
                f"Cannot initialize a Ket: purity is {p:.5f} (must be at least 1.0-{atol_purity}).",
            )
        return Ket.from_ansatz(
            modes,
            coeff * PolyExpAnsatz.from_function(fn=wigner_to_bargmann_psi, cov=cov, means=means),
            name,
        )

    @classmethod
    def random(cls, modes: Collection[int], max_r: float = 1.0) -> Ket:
        r"""
        Generates a random zero displacement state.

        Args:
            modes: The modes of the state.
            max_r: Maximum squeezing parameter over which we make random choices.
        Output is a Ket
        """

        m = len(modes)
        S = math.random_symplectic(m, max_r)
        transformation = (
            1
            / math.sqrt(complex(2))
            * math.block(
                [
                    [
                        math.eye(m, dtype=math.complex128),
                        math.eye(m, dtype=math.complex128),
                    ],
                    [
                        -1j * math.eye(m, dtype=math.complex128),
                        1j * math.eye(m, dtype=math.complex128),
                    ],
                ]
            )
        )
        S = math.conj(math.transpose(transformation)) @ S @ transformation
        S_1 = S[:m, :m]
        S_2 = S[:m, m:]
        A = math.transpose(math.solve(math.dagger(S_1), math.transpose(S_2)))
        b = math.zeros(m, dtype=A.dtype)
        psi = cls.from_bargmann(modes, (A, b, complex(1)))
        return psi.normalize()

    def auto_shape(
        self, max_prob=None, max_shape=None, respect_manual_shape=True
    ) -> tuple[int, ...]:  # TODO: revisit
        r"""
        A good enough estimate of the Fock shape of this Ket, defined as the shape of the Fock
        array (batch excluded) if it exists, and if it doesn't exist it is computed as the shape
        that captures at least ``settings.AUTOSHAPE_PROBABILITY`` of the probability mass of each
        single-mode marginal (default 99.9%).
        If the ``respect_manual_shape`` flag is set to ``True``, auto_shape will respect the
        non-None values in ``manual_shape``.

        Args:
            max_prob: The maximum probability mass to capture in the shape (default from ``settings.AUTOSHAPE_PROBABILITY``).
            max_shape: The maximum shape to compute (default from ``settings.AUTOSHAPE_MAX``).
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``.
        """
        if self.ansatz.batch_size <= 1:
            try:  # fock
                shape = self.ansatz.core_shape
            except AttributeError:  # bargmann
                if self.ansatz.num_derived_vars == 0:
                    ansatz = self.ansatz.conj & self.ansatz
                    A, b, c = (
                        (ansatz.A[0], ansatz.b[0], ansatz.c[0])
                        if ansatz.batch_shape != ()  # tensorflow
                        else ansatz.triple
                    )
                    ansatz = ansatz / self.probability
                    shape = autoshape_numba(
                        math.asnumpy(A),
                        math.asnumpy(b),
                        math.asnumpy(c),
                        max_prob or settings.AUTOSHAPE_PROBABILITY,
                        max_shape or settings.AUTOSHAPE_MAX,
                    )
                else:
                    shape = [settings.AUTOSHAPE_MAX] * len(self.modes)
        else:
            warnings.warn("auto_shape only looks at the shape of the first element of the batch.")
            shape = [settings.AUTOSHAPE_MAX] * len(self.modes)
        if respect_manual_shape:
            return tuple(c or s for c, s in zip(self.manual_shape, shape))
        return tuple(shape)

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``Ket``.
        """
        # TODO: assuming first batch dim is lin_sup
        if self.ansatz._lin_sup:
            str1 = generate_batch_str(self.ansatz.batch_shape)
            str2 = str1[:-1] + chr(ord(str1[-1]) + 1)
            mode = f"{str1},{str2}->{str1}{str2[-1]}"
        else:
            mode = "zip"
        repr = self.representation.contract(self.adjoint.representation, mode=mode)

        # TODO: assuming first batch dim is lin_sup
        if self.ansatz._lin_sup:
            A, b, c = repr.ansatz.triple
            batch_shape = self.ansatz.batch_shape[:-1]
            new_A = math.reshape(A, batch_shape + (-1,) + tuple(A.shape[-2:]))
            new_b = math.reshape(b, batch_shape + (-1,) + tuple(b.shape[-1:]))
            new_c = math.reshape(c, batch_shape + (-1,) + self.ansatz.shape_derived_vars)
            new_ansatz = PolyExpAnsatz(new_A, new_b, new_c, lin_sup=True)
            repr._ansatz = new_ansatz

        ret = DM(repr, self.name)
        ret.manual_shape = self.manual_shape + self.manual_shape
        return ret

    def expectation(self, operator: CircuitComponent):  # TODO: revisit
        r"""
        The expectation value of an operator calculated with respect to this Ket.

        Given the operator `O`, this function returns :math:`Tr\big(|\psi\rangle\langle\psi| O)`\,
        where :math:`|\psi\rangle` is the vector representing this state.

        The ``operator`` is expected to be a component with ket-like wires (i.e., output wires on
        the ket side), density matrix-like wires (output wires on both ket and bra sides), or
        unitary-like wires (input and output wires on the ket side).

        Args:
            operator: A ket-like, density-matrix like, or unitary-like circuit component.

        Raise:
            ValueError: If ``operator`` is not a ket-like, density-matrix like, or unitary-like
                component.
            ValueError: If ``operator`` is defined over a set of modes that is not a subset of the
                modes of this state.
        """
        if (self.ansatz and self.ansatz.batch_shape != ()) or (
            operator.ansatz and operator.ansatz.batch_shape != ()
        ):
            raise NotImplementedError("Batched expectation values are not implemented.")

        op_type, msg = _validate_operator(operator)
        if op_type is OperatorType.INVALID_TYPE:
            raise ValueError(msg)

        if not operator.wires.modes.issubset(self.wires.modes):
            msg = f"Expected an operator defined on a subset of modes `{self.modes}`, "
            msg += f"found one defined on `{operator.modes}.`"
            raise ValueError(msg)

        leftover_modes = self.wires.modes - operator.wires.modes
        if op_type is OperatorType.KET_LIKE:
            result = self.contract(operator.dual)
            result = result.contract(result.adjoint)
            result >>= TraceOut(leftover_modes)

        elif op_type is OperatorType.DM_LIKE:
            result = (self.adjoint.contract(self.contract(operator.dual))) >> TraceOut(
                leftover_modes
            )

        else:
            result = (self.contract(operator)) >> self.dual

        return result

    def _ipython_display_(self):  # pragma: no cover
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        is_fock = isinstance(self.ansatz, ArrayAnsatz)
        display(widgets.state(self, is_ket=True, is_fock=is_fock))

    def __getitem__(self, idx: int | Sequence[int]) -> State:
        r"""
        Reduced density matrix obtained by tracing out all the modes except those in
        ``idx``. Note that the result is returned with modes in increasing order.

        Args:
            idx: The modes to keep.

        Returns:
            A ``DM`` object with the remaining modes.
        """
        return self.dm()[idx]

    def __rshift__(self, other: CircuitComponent | Scalar) -> CircuitComponent | Batch[Scalar]:
        r"""
        Contracts ``self`` and ``other`` (output of self into the inputs of other),
        adding the adjoints when they are missing. Given this is a ``Ket`` object which
        has only ket wires at the output, in expressions like ``ket >> channel`` where ``channel``
        has wires on the ket and bra sides the adjoint of ket is automatically added, effectively
        calling ``ket.adjoint @ (ket @ channel)`` and the method returns a new ``DM``.
        In expressions lke ``ket >> u`` where ``u`` is a unitary, the adjoint of ``ket`` is
        not needed and the method returns a new ``Ket``.

        Returns a ``DM`` or a ``Ket`` when the wires of the resulting components are compatible
        with those of a ``DM`` or of a ``Ket``. Returns a ``CircuitComponent`` in general,
        and a (batched) scalar if there are no wires left, for convenience.
        """
        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input:
            if not result.wires.bra:
                return Ket(result.representation)
            elif result.wires.bra.modes == result.wires.ket.modes:
                return DM(result.representation)
        return result
