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

from collections.abc import Collection, Sequence

import numpy as np
from IPython.display import display

from mrmustard import math, settings, widgets
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_psi
from mrmustard.physics.gaussian import purity
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexVector, Scalar

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..transformations import Operation, Unitary
from ..utils import shape_check
from .base import OperatorType, State, _validate_operator
from .dm import DM

__all__ = ["Ket"]


class Ket(State):
    r"""
    Base class for all Hilbert space vectors.
    """

    short_name = "Ket"

    @property
    def is_physical(self) -> bool:
        r"""
        Whether the ket object is a physical one.

        Returns:
            A boolean variable.

        Raises:
            NotImplementedError: If the state is in a linear superposition.
            NotImplementedError: If the state has derived variables.
            NotImplementedError: If the state has an ``ArrayAnsatz``.

        Example:
        .. code-block::

            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0])

            >>> assert psi.is_physical
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError(
                "Physicality conditions are not implemented for a linear superposition of states.",
            )
        if self.ansatz.num_derived_vars > 0:
            raise NotImplementedError(
                "Physicality conditions are not implemented for derived variables.",
            )
        if isinstance(self.ansatz, ArrayAnsatz):
            raise NotImplementedError(
                "Physicality conditions are not implemented for states with ArrayAnsatz.",
            )
        return math.all(math.abs(math.eigvals(self.ansatz.A)) < 1) and math.allclose(
            self.probability,
            1,
            settings.ATOL,
        )

    @property
    def probability(self) -> float:
        r"""
        Probability of this Ket (L2 norm squared).

        Returns:
            The probability of this ``Ket``.

        Example:
        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0])

            >>> assert math.allclose(psi.probability, 1.0)
        """
        return self.L2_norm

    @property
    def purity(self) -> float:
        r"""
        The purity of the state.

        Returns:
            The purity of this ``Ket`` (always 1.0).

        Example:
        .. code-block::

            >>> from mrmustard.lab import Ket
            >>> assert Ket.random([0]).purity == 1.0
        """
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
        if ansatz and ansatz.core_dims != len(modes):
            raise ValueError(
                f"Expected an ansatz with {len(modes)} variables, found {ansatz.core_dims}.",
            )
        wires = Wires(modes_out_ket=modes)
        return Ket(ansatz, wires, name=name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
        atol_purity: float | None = None,
    ) -> Ket:
        cov, means, coeff = triple
        cov = math.astensor(cov)
        means = math.astensor(means)
        if cov.shape[:-2] != ():  # pragma: no cover
            raise NotImplementedError("Not implemented for batched states.")
        shape_check(cov, means, 2 * len(modes), "Phase space")
        if atol_purity:
            p = math.cast(purity(cov), math.float64)
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

        Returns:
            A ``Ket`` object.

        .. details::
            The output is a random Gaussian unitary :math:`U` over :math:`modes` with
            zero displacement and maximum squeezing `max_r` applied to vacuum state over `modes`,
            i.e., :math:`U|0^{modes}\rangle`.

        Example:
        .. code-block::

            >>> from mrmustard.lab import Ket
            >>> assert isinstance(Ket.random([0,1]), Ket)
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
                ],
            )
        )
        S = math.conj(math.transpose(transformation)) @ S @ transformation
        S_1 = S[:m, :m]
        S_2 = S[:m, m:]
        A = math.transpose(math.solve(math.dagger(S_1), math.transpose(S_2)))
        b = math.zeros(m, dtype=A.dtype)
        psi = cls.from_bargmann(modes, (A, b, complex(1)))
        return psi.normalize()

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``Ket``.

        Returns:
            A ``DM``.

        .. code-block::

            >>> from mrmustard.lab import Vacuum, DM
            >>> assert isinstance(Vacuum([0]).dm(), DM)
        """
        ret = self.contract(self.adjoint, mode="zip")
        return DM._from_attributes(ret.ansatz, ret.wires, name=self.name)

    def expectation(self, operator: CircuitComponent, mode: str = "kron") -> Batch[Scalar]:
        r"""
        The expectation value of an operator calculated with respect to this Ket.

        Args:
            operator: A ket-like, density-matrix like, or unitary-like circuit component.
            mode: The mode of contraction. Can either "zip" the batch dimensions, "kron" the batch dimensions,
                or pass a custom einsum-style batch string like "ab,cb->ac".

        Returns:
            Expectation value as a complex number.

        Raises:
            ValueError: If ``operator`` is not a ket-like, density-matrix like, or unitary-like
                component.
            ValueError: If ``operator`` is defined over a set of modes that is not a subset of the
                modes of this state.

        Note:
            Given the operator `O`, this function returns :math:`Tr\big(|\psi\rangle\langle\psi| O)`\,
            where :math:`|\psi\rangle` is the vector representing this state.

            The ``operator`` is expected to be a component with ket-like wires (i.e., output wires on
            the ket side), density matrix-like wires (output wires on both ket and bra sides), or
            unitary-like wires (input and output wires on the ket side).

        Example:

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Number, Rgate

            >>> psi = Number(0, 1)
            >>> theta = 0.123
            >>> answer = math.exp(1j*theta)

            >>> assert math.allclose(psi.expectation(Rgate(0, theta)), answer)
        """
        op_type, msg = _validate_operator(operator)
        if op_type is OperatorType.INVALID_TYPE:
            raise ValueError(msg)

        if not operator.wires.modes.issubset(self.wires.modes):
            msg = f"Expected an operator defined on a subset of modes `{self.modes}`, "
            msg += f"found one defined on `{operator.modes}.`"
            raise ValueError(msg)

        leftover_modes = self.wires.modes - operator.wires.modes
        if op_type is OperatorType.KET_LIKE:
            result = self.contract(operator.dual, mode=mode)
            result = result.contract(result.adjoint, mode="zip") >> TraceOut(leftover_modes)
        elif op_type is OperatorType.DM_LIKE:
            result = self.adjoint.contract(
                self.contract(operator.dual, mode=mode),
                mode="zip",
            ) >> TraceOut(leftover_modes)
        else:
            result = (self.contract(operator, mode=mode)).contract(self.dual, mode="zip")
            result = result >> TraceOut(result.modes)
        return result

    def fidelity(self, other: State) -> float:
        r"""
        The fidelity between this ket and another state.

        .. details::

            .. math::
                F(|\psi\rangle, \phi\rangle) = |\langle \psi, \phi \rangle|^2

        """
        if self.modes != other.modes:
            raise ValueError("Cannot compute fidelity between states with different modes.")
        return self.expectation(other)

    def formal_stellar_decomposition(self, core_modes):
        r"""
        Applies the formal stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            S: The core state (`Ket`).
            T: The Gaussian `Operation` performing the stellar decomposition.

        Note:
            This method pulls out the unitary ``U`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \psi = (T\otimes\mathbb I) S_{\mathrm{core}}

            where the operator :math:`T` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.

        .. code-block::

            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0,1])
            >>> core, t = psi.formal_stellar_decomposition([0])
            >>> A_core = core.ansatz.A

            >>> assert A_core[0,0] == 0
        """
        other_modes = [m for m in self.modes if m not in core_modes]
        core_indices = self.wires[core_modes].indices
        other_indices = self.wires[other_modes].indices
        new_order = core_indices + other_indices

        A, b, c = self.ansatz.reorder(new_order).triple

        M = len(core_modes)

        # we pick the blocks according to the naming chosen in the paper
        Am = A[..., :M, :M]
        R = A[..., M:, :M]
        R_transpose = math.einsum("...ij->...ji", R)
        An = A[..., M:, M:]
        bm = b[..., :M]
        bn = b[..., M:]

        batch_shape = self.ansatz.batch_shape

        Om = math.zeros((*batch_shape, M, M), dtype=math.complex128)
        As = math.block([[Om, R_transpose], [R, An]])

        bs = math.concat([math.zeros((*batch_shape, M), dtype=math.complex128), bn], -1)
        cs = c

        inverse_order = np.argsort(new_order)
        As = As[..., inverse_order, :]
        As = As[..., :, inverse_order]
        bs = bs[..., inverse_order]

        s = Ket.from_bargmann(self.modes, (As, bs, cs))

        if batch_shape != ():
            Im = math.stack(
                [math.eye(M, dtype=math.complex128)] * int(math.prod(batch_shape)),
            ).reshape(batch_shape + (M,) * 2)
        else:
            Im = math.eye(M, dtype=math.complex128)

        At = math.block([[Am, Im], [Im, Om]])

        bt = math.concat([bm, math.zeros((*batch_shape, M), dtype=math.complex128)], -1)
        ct = math.ones_like(c)
        t = Operation.from_bargmann(core_modes, core_modes, (At, bt, ct))

        return s, t

    def physical_stellar_decomposition(self, core_modes):
        r"""
        Applies the physical stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            psi_core: The core state (`Ket`)
            U: The Gaussian unitary performing the stellar decomposition.

        Note:
            This method pulls out the unitary ``U`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \psi = (U\otimes\mathbb I) \psi_{\mathrm{core}}

            where the unitary :math:`U` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket

            >>> psi = Ket.random([0,1])
            >>> core, U = psi.physical_stellar_decomposition([0])
            >>> assert psi == core >> U

            >>> A_c = core.ansatz.A
            >>> assert math.allclose(A_c[0,0], 0)
        """
        # bringing A to the ordering of our interest
        other_modes = [m for m in self.modes if m not in core_modes]
        core_indices = self.wires[core_modes].indices
        other_indices = self.wires[other_modes].indices
        new_order = core_indices + other_indices

        A, b, c = self.ansatz.reorder(new_order).triple
        M = len(core_modes)

        # we pick the blocks according to the naming chosen in the paper
        Am = A[..., :M, :M]
        An = A[..., M:, M:]
        R = A[..., M:, :M]
        bm = b[..., :M]
        bn = b[..., M:]

        batch_shape = self.ansatz.batch_shape

        gamma_squared = math.eye(M, dtype=math.complex128) - Am @ math.conj(Am)
        gamma_evals, gamma_evecs = math.eigh(gamma_squared)

        gamma = math.einsum(
            "...ij,...j,...kj->...ik",
            gamma_evecs,
            math.sqrt(gamma_evals),
            math.conj(gamma_evecs),
        )
        gamma_transpose = math.einsum("...ij->...ji", gamma)
        gamma_inv = math.inv(gamma)
        gamma_inv_T = math.einsum("...ij->...ji", gamma_inv)

        Au = math.block([[Am, gamma], [gamma_transpose, -math.conj(Am)]])

        bu_in = -math.einsum(
            "...ij,...jk,...k->...i",
            math.conj(Am),
            gamma_inv_T,
            bm,
        ) - math.einsum("...ij,...j->...i", gamma_inv, math.conj(bm))
        bu = math.concat([bm, bu_in], -1)
        cu = math.ones(batch_shape, dtype=math.complex128)
        U = Unitary.from_bargmann(core_modes, core_modes, (Au, bu, cu))

        u_renorm = (U.contract(U.dual, mode="zip")).ansatz.c

        U /= math.sqrt(u_renorm)

        R_T = math.einsum("...ij->...ji", R)
        A_core = math.block(
            [
                [math.zeros((*batch_shape, M, M), dtype=math.complex128), gamma_inv @ R_T],
                [R @ gamma_inv_T, An + R @ math.inv(math.inv(math.conj(Am)) - Am) @ R_T],
            ],
        )

        Rc = R @ gamma_inv_T
        b_tmp = bn - math.einsum("...ij,...j->...i", Rc, bu_in)
        b_core = math.concat([math.zeros_like(bm), b_tmp], -1)

        inverse_order = np.argsort(new_order)
        A_core = A_core[..., inverse_order, :][..., :, inverse_order]
        b_core = b_core[..., inverse_order]
        c_core = c / U.ansatz.c
        core = Ket.from_bargmann(self.modes, (A_core, b_core, c_core))

        return core, U

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
        adding the adjoints when they are missing.

        Args:
            other: the ``CircuitComponent`` object that we want to contract the state with.

        Returns:
            A ``DM`` or a ``Ket`` when the wires of the resulting components are compatible
            with those of a ``DM`` or of a ``Ket``. Returns a ``CircuitComponent`` in general,
            and a (batched) scalar if there are no wires left, for convenience.

        Note:
            Given this is a ``Ket`` object which
            has only ket wires at the output, in expressions like ``ket >> channel`` where ``channel``
            has wires on the ket and bra sides the adjoint of ket is automatically added, effectively
            calling ``ket.adjoint @ (ket @ channel)`` and the method returns a new ``DM``.
            In expressions lke ``ket >> u`` where ``u`` is a unitary, the adjoint of ``ket`` is
            not needed and the method returns a new ``Ket``.

        .. code-block::

            >>> from mrmustard.lab import Ket, DM, Attenuator, Dgate

            >>> psi = Ket.random([0,1])
            >>> U = Dgate(0, x=1, y=0)
            >>> channel = Attenuator(0, .5)

            >>> assert isinstance(psi >> U, Ket)
            >>> assert isinstance(psi >> channel, DM)
        """
        result = super().__rshift__(
            other,
        )  # this would be the output if we didn't override __rshift__
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        # TODO: Reminder: replace with result.wires.ket_like and result.wires.dm_like
        if not result.wires.input:
            if not result.wires.bra:
                return Ket._from_attributes(result.ansatz, result.wires)
            if result.wires.bra.modes == result.wires.ket.modes:
                return DM._from_attributes(result.ansatz, result.wires)
        return result
