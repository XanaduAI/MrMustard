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

"""This module contains the defintion of the ket class ``Ket``."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence

import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt

from mrmustard import math, settings, widgets
from mrmustard.physics import stellar
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.gaussian import purity
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import ComplexTensor, Matrix, Scalar, Vector

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..transformations import Operation, Unitary
from ..utils import shape_check
from .base import State
from .builtins import wigner_to_bargmann_psi
from .dm import DM

__all__ = ["Ket"]


class Ket(State):
    r"""Base class for all Hilbert space vectors."""

    short_name = "Ket"

    @property
    def is_physical(self) -> bool:
        r"""Whether the ket object is a physical one.

        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0])
        >>> assert psi.is_physical

        Returns:
            A boolean variable.

        Raises:
            NotImplementedError: If the state is in a linear superposition.
            NotImplementedError: If the state has derived variables.
            NotImplementedError: If the state has an ``ArrayAnsatz``.
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
        r"""Probability of this Ket (L2 norm squared).

        >>> from mrmustard import math
        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0])
        >>> assert math.allclose(psi.probability, 1.0)

        Returns:
            The probability of this ``Ket``.
        """
        return self.L2_norm

    @property
    def purity(self) -> float:
        r"""The purity of the state.

        >>> from mrmustard.lab import GaussianKet
        >>> assert GaussianKet.random([0]).purity == 1.0

        Returns:
            The purity of this ``Ket`` (always 1.0).
        """
        try:
            shape = (
                self.ansatz.batch_shape[:-1] if self.ansatz._lin_sup else self.ansatz.batch_shape
            )
        except AttributeError as e:
            if str(e) != "CircuitComponent has no ansatz factory.":
                raise e
            shape = ()
        return math.ones(shape)

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        name: str | None = None,
        lin_sup: bool = False,
    ) -> Ket:
        return super().from_bargmann(modes=modes, triple=triple, name=name, lin_sup=lin_sup)

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batch_dims: int = 0,
    ) -> Ket:
        return super().from_fock(modes=modes, array=array, name=name, batch_dims=batch_dims)

    @classmethod
    def from_ansatz(
        cls,
        modes: Collection[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Ket:
        if not isinstance(modes, set) and sorted(modes) != list(modes):
            raise ValueError(f"Modes must be sorted. Got {modes}")
        modes = set(modes)
        if ansatz and ansatz.core_dims != len(modes):
            raise ValueError(
                f"Expected an ansatz with {len(modes)} variables, found {ansatz.core_dims}.",
            )
        wires = Wires(modes_out_ket=modes)
        if ansatz is not None:
            ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz)
            if representation == ReprEnum.FOCK:
                for w in wires.quantum:
                    w.repr = ReprEnum.FOCK
                    w.fock_shape = ansatz.core_shape[w.index]
        else:
            ansatz_factory = None
        return Ket(ansatz_factory=ansatz_factory, wires=wires, name=name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple[Matrix, Vector, Scalar],
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

        ansatz_dict = {
            ReprEnum.BARGMANN: (wigner_to_bargmann_psi, ("cov", "means", "lin_sup")),
        }
        ansatz_factory = AnsatzFactory(ansatz_dict=ansatz_dict, cov=cov, means=means)
        ansatz = ansatz_factory(cov=cov, means=means, representation=ReprEnum.BARGMANN)
        return Ket.from_ansatz(
            modes,
            coeff * ansatz,
            name,
        )

    @classmethod
    def from_quadrature(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        phi: float = 0.0,
        name: str | None = None,
    ) -> Ket:
        return super().from_quadrature(modes=modes, triple=triple, phi=phi, name=name)

    def dm(self) -> DM:
        r"""The ``DM`` object obtained from this ``Ket``.

        >>> from mrmustard.lab import Vacuum, DM
        >>> assert isinstance(Vacuum([0]).dm(), DM)

        Returns:
            A ``DM``.
        """
        ret = self.contract(self.adjoint)
        return DM._from_attributes(ret.ansatz, ret.wires, name=self.name)

    def expectation(self, operator: CircuitComponent) -> Scalar:
        r"""The expectation value of an operator calculated with respect to this Ket.

        >>> from mrmustard import math
        >>> from mrmustard.lab import Number, Rgate
        >>> psi = Number(0, 1)
        >>> theta = 0.123
        >>> answer = math.exp(1j*theta)
        >>> assert math.allclose(psi.expectation(Rgate(0, theta)), answer)

        Args:
            operator: A ket-like, density-matrix like, or unitary-like circuit component.

        Returns:
            Expectation value as a complex number.

        Raises:
            ValueError: If the operator is not a ket-like, density-matrix like, or unitary-like component.
            ValueError: If the operator is defined over a set of modes that is not a subset of the modes of this state.
            ValueError: If the modes in common between the operator and the state are not a subset of the modes of the operator.

        Note:
            Given the operator `O`, this function returns :math:`Tr\big(|\psi\rangle\langle\psi| O)`\,
            where :math:`|\psi\rangle` is the vector representing this state.

            The ``operator`` is expected to be a component with ket-like wires (i.e., output wires on
            the ket side), density matrix-like wires (output wires on both ket and bra sides), or
            unitary-like wires (input and output wires on the ket side).
        """
        if operator.wires.is_ket_like:
            ret = self.contract(operator.dual)
            if isinstance(ret.ansatz, PolyExpAnsatz) and ret.ansatz.num_derived_vars > 0:
                ret = ret.to_fock()
            return math.real(ret.contract(ret.dual).ansatz.scalar)

        if operator.wires.is_dm_like:
            operator >>= TraceOut(operator.wires.modes - self.wires.modes)
            self >>= TraceOut(self.wires.modes - operator.wires.modes)
            ret = operator.expectation(self.dm())

        elif operator.wires.is_unitary_like:
            if not operator.wires.modes.issubset(self.wires.modes):
                raise ValueError(
                    f"Operator modes {operator.wires.modes} are not "
                    f"a subset of the state modes {self.wires.modes}"
                )
            self >>= TraceOut(self.wires.modes - operator.wires.modes)
            ret = self.dm().contract(operator) >> TraceOut(operator.wires.modes)

        else:
            raise ValueError(
                f"Cannot calculate the expectation value of {operator} with respect to {self}."
            )

        if isinstance(ret, CircuitComponent) and len(ret.wires.modes) == 0:
            return ret.ansatz.scalar
        return ret

    def fidelity(self, other: State) -> Scalar:
        r"""The fidelity between this ket and another state.

        .. details::

            .. math::
                F(|\psi\rangle, \phi\rangle) = |\langle \psi, \phi \rangle|^2

        Args:
            other: The other state.

        Returns:
            The fidelity between this ket and the other state.

        Raises:
            ValueError: If the states have different modes.
        """
        if self.modes != other.modes:
            raise ValueError("Cannot compute fidelity between states with different modes.")
        return self.expectation(other)

    def formal_stellar_decomposition(self, core_modes: Collection[int]) -> tuple[Ket, Operation]:
        r"""Applies the formal stellar decomposition.

        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0,1])
        >>> core, t = psi.formal_stellar_decomposition([0])
        >>> A_core = core.ansatz.A
        >>> assert A_core[0,0] == 0

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            The core state (`Ket`) and the Gaussian `Operation` performing the stellar decomposition.

        Note:
            This method pulls out the unitary ``U`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \psi = (T\otimes\mathbb I) S_{\mathrm{core}}

            where the operator :math:`T` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.
        """
        other_modes = [m for m in self.modes if m not in core_modes]
        idx = self.wires[core_modes].indices + self.wires[other_modes].indices
        inv = np.argsort(idx)

        A, b, c = self.bargmann_triple()
        A = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2)
        b = math.gather(b, idx, axis=-1)

        core_tr, op_tr = stellar.formal_stellar_triples((A, b, c), len(core_modes))

        A_core = math.gather(math.gather(core_tr[0], inv, axis=-1), inv, axis=-2)
        b_core = math.gather(core_tr[1], inv, axis=-1)

        core = Ket.from_bargmann(self.modes, (A_core, b_core, core_tr[2]))
        t = Operation.from_bargmann(core_modes, core_modes, op_tr)

        return core, t

    def physical_stellar_decomposition(self, core_modes: Collection[int]) -> tuple[Ket, Unitary]:
        r"""Applies the physical stellar decomposition.

        >>> from mrmustard import math
        >>> from mrmustard.lab import GaussianKet
        >>> psi = GaussianKet.random([0,1])
        >>> core, U = psi.physical_stellar_decomposition([0])
        >>> assert psi == core >> U
        >>> A_c = core.ansatz.A
        >>> assert math.allclose(A_c[0,0], 0)

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            The core state (`Ket`) and the Gaussian unitary performing the stellar decomposition.

        Note:
            This method pulls out the unitary `U` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have

            .. math::

                \psi = (U\otimes\mathbb I) \psi_{\mathrm{core}}

            where the unitary `U` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation e.g., being sparse.
        """
        other_modes = [m for m in self.modes if m not in core_modes]
        idx = self.wires[core_modes].indices + self.wires[other_modes].indices
        inv = np.argsort(idx)

        A, b, c = self.bargmann_triple()
        A = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2)
        b = math.gather(b, idx, axis=-1)

        core_tr, u_tr = stellar.physical_stellar_triples_ket((A, b, c), len(core_modes))

        A_core = math.gather(math.gather(core_tr[0], inv, axis=-1), inv, axis=-2)
        b_core = math.gather(core_tr[1], inv, axis=-1)

        core = Ket.from_bargmann(self.modes, (A_core, b_core, core_tr[2]))
        U = Unitary.from_bargmann(core_modes, core_modes, u_tr)

        return core, U

    def stellar_roots(self, max_degree: int | None = None) -> np.ndarray:
        r"""Compute the stellar roots of this single-mode ket.

        The stellar roots are the zeros of the Bargmann polynomial

        .. math::

            F(z) = \sum_n  c_n z^n / \sqrt{n!}

        built from the Fock amplitudes :math:`c_{n}` of this state.

        For states already in Fock representation the amplitudes are used
        directly.  For Bargmann states whose ``c`` tensor is scalar (i.e. pure
        Gaussian states with no polynomial part) there are no finite roots and
        an empty array is returned.  For Bargmann states with a non-scalar
        ``c`` the state is converted to Fock using the polynomial degree
        (or the explicit ``max_degree`` if provided, clamped to the true
        polynomial degree to avoid spurious roots from the Gaussian envelope).
        The polynomial degree is ``sum(n_i - 1)`` where ``n_i`` are the
        dimensions of the derived-variable part of ``c``.

        Args:
            max_degree: Maximum polynomial degree for computing the roots.
                For ``PolyExpAnsatz`` states this is clamped to the true
                polynomial degree.  If ``None``, the degree is inferred
                from the ansatz.

        Returns:
            Complex 1D array of stellar roots (empty for Gaussian states).

        Raises:
            ValueError: If the state has more than one mode or is batched.
        """
        if self.n_modes != 1:
            raise ValueError(
                f"Stellar roots are defined for single-mode kets, "
                f"but this state has {self.n_modes} modes."
            )
        if self.ansatz._lin_sup:
            raise ValueError(
                "Stellar roots are not directly supported for linear "
                "superpositions.  Convert to Fock first with a large enough "
                "cutoff, e.g.  state.to_fock(cutoff).stellar_roots()"
            )
        if self.ansatz.batch_shape:
            raise ValueError(
                f"Stellar roots are not supported for batched states "
                f"(batch shape {self.ansatz.batch_shape})."
            )

        if isinstance(self.ansatz, ArrayAnsatz):
            fock_shape = [max_degree + 1] if max_degree is not None else None
            return stellar.stellar_roots(np.asarray(self.fock_array(fock_shape)).ravel())

        # PolyExpAnsatz: check whether c carries a polynomial part
        if not self.ansatz.shape_derived_vars:
            return np.array([], dtype=np.complex128)

        # Non-scalar c: each derived-variable dimension n_i contributes max
        # degree n_i - 1, so the total polynomial degree is sum(n_i - 1).
        # Clamp max_degree to this value: Fock amplitudes beyond the true
        # polynomial degree carry Gaussian-envelope contributions that would
        # introduce spurious roots.
        degree = sum(n - 1 for n in self.ansatz.shape_derived_vars)
        if max_degree is not None:
            if max_degree > degree:
                warnings.warn(
                    f"max_degree={max_degree} exceeds the true polynomial degree {degree}. "
                    f"Clamping to {degree} to avoid spurious roots from Gaussian envelope contributions.",
                    stacklevel=2,
                )
            degree = min(degree, max_degree)
        return stellar.stellar_roots(np.asarray(self.fock_array([degree + 1])).ravel())

    def plot_stellar_roots(
        self,
        max_degree: int | None = None,
        *,
        ax: plt.Axes | None = None,
        max_limit: float = 10.0,
    ) -> plt.Axes:
        """Plot the stellar roots of this single-mode ket.

        Each root is plotted in the complex plane and colored by its phase
        angle via the HSV colormap.

        Args:
            max_degree: Maximum polynomial degree for computing the roots.
                If ``None``, the degree is inferred from the ansatz.
            ax: Optional matplotlib ``Axes`` to draw on.  When provided the
                roots are drawn on the existing axes (no new figure is
                created and ``plt.show()`` is not called).  Useful with the
                ipympl backend for flicker-free interactive updates.
            max_limit: Maximum absolute value for each axis.

        Returns:
            The matplotlib ``Axes`` instance used for the plot.

        Raises:
            ValueError: If the state has more than one mode or is batched.
        """
        return stellar.plot_stellar_roots(
            self.stellar_roots(max_degree),
            ax=ax,
            max_limit=max_limit,
        )

    def _ipython_display_(self):  # pragma: no cover
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        is_fock = isinstance(self.ansatz, ArrayAnsatz)
        display(widgets.state(self, is_ket=True, is_fock=is_fock))

    def __rshift__(self, other: Scalar | CircuitComponent) -> Scalar | CircuitComponent:
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
