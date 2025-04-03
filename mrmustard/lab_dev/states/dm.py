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
This module contains the defintion of the density matrix class ``DM``.
"""

from __future__ import annotations
from typing import Collection, Sequence

import warnings
import numpy as np
from IPython.display import display

from mrmustard import math, settings, widgets
from mrmustard.math.lattice.strategies.vanilla import autoshape_numba
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_rho
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from mrmustard.physics.representations import Representation
from mrmustard.physics.wires import Wires, ReprEnum
from mrmustard.utils.typing import ComplexTensor, RealVector

from .base import State, _validate_operator, OperatorType
from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..transformations import Map, Channel

from ..utils import shape_check

__all__ = ["DM"]


class DM(State):
    r"""
    Base class for density matrices.
    """

    short_name = "DM"

    @property
    def is_positive(self) -> bool:  # TODO: revisit this
        r"""
        Whether this DM is a positive operator.
        """
        batch_dim = self.ansatz.batch_size
        if batch_dim > 1:
            raise ValueError(
                "Physicality conditions are not implemented for batch dimension larger than 1."
            )
        if self.ansatz.num_derived_vars > 0:
            raise ValueError("Physicality conditions are not implemented for derived variables.")
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[..., :m, m:]

        if (
            math.real(math.norm(gamma_A - math.conj(gamma_A.T))) > settings.ATOL
        ):  # checks if gamma_A is Hermitian
            return False

        return all(math.real(math.eigvals(gamma_A)) >= 0)

    @property
    def is_physical(self) -> bool:
        r"""
        Whether this DM is a physical density operator.
        """
        return self.is_positive and math.allclose(self.probability, 1, settings.ATOL)

    @property
    def probability(self) -> float:
        r"""
        Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states.
        """
        return math.sum(self._probabilities)

    @property
    def purity(self) -> float:
        return self.L2_norm

    @property
    def _probabilities(self) -> RealVector:
        r"""
        Element-wise probabilities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states.
        """
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices
        rep = self.ansatz.trace(idx_ket, idx_bra)
        return math.real(math.sum(rep.scalar))

    @property
    def _purities(self) -> RealVector:
        r"""
        Element-wise purities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states.
        """
        return self._L2_norms / self._probabilities

    @classmethod
    def from_ansatz(
        cls,
        modes: Collection[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> State:
        if not isinstance(modes, set) and sorted(modes) != list(modes):
            raise ValueError(f"Modes must be sorted. got {modes}")
        modes = set(modes)
        if ansatz and ansatz.num_vars != 2 * len(modes):
            raise ValueError(
                f"Expected an ansatz with {2*len(modes)} variables, found {ansatz.num_vars}."
            )
        wires = Wires(modes_out_bra=set(modes), modes_out_ket=set(modes))
        if isinstance(ansatz, ArrayAnsatz):
            for w in wires:
                w.repr = ReprEnum.FOCK
        return DM(Representation(ansatz, wires), name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple,
        name: str | None = None,
        s: float = 0,  # pylint: disable=unused-argument
    ) -> DM:
        r"""
        Initializes a density matrix from the covariance matrix, vector of means and a coefficient,
        which parametrize the s-parametrized phase space function
        :math:`coeff * exp(-1/2(x-means)^T cov^{-1} (x-means))`.h:`coeff * exp((x-means)^T cov^{-1} (x-means))`.


        Args:
            modes: The modes of this states.
            triple: The ``(cov, means, coeff)`` triple.
            name: The name of this state.
            s: The phase space parameter, defaults to 0 (Wigner).
        """
        cov, means, coeff = triple
        cov = math.astensor(cov)
        means = math.astensor(means)
        shape_check(cov, means, 2 * len(modes), "Phase space")
        return coeff * DM.from_ansatz(
            modes,
            PolyExpAnsatz.from_function(fn=wigner_to_bargmann_rho, cov=cov, means=means),
            name,
        )

    @classmethod
    def random(cls, modes: Collection[int], m: int | None = None, max_r: float = 1.0) -> DM:
        r"""
        Returns a random ``DM`` with zero displacement.

        Args:
            modes: The modes of the ``DM``.
            m: The number modes to be considered for tracing out from a random pure state (Ket)
                if not specified, m is considered to be len(modes)
        """
        if m is None:
            m = len(modes)
        max_idx = max(modes)
        ancilla = list(range(max_idx + 1, max_idx + m + 1))
        full_modes = list(modes) + ancilla
        m = len(full_modes)
        S = math.random_symplectic(m, max_r)
        I = math.eye(m, dtype=math.complex128)
        transformation = math.block([[I, I], [-1j * I, 1j * I]]) / np.sqrt(2)
        S = math.conj(math.transpose(transformation)) @ S @ transformation
        S_1 = S[:m, :m]
        S_2 = S[:m, m:]
        A = math.transpose(math.solve(math.dagger(S_1), math.transpose(S_2)))
        b = math.zeros(m, dtype=A.dtype)
        A, b, c = complex_gaussian_integral_2(
            (math.conj(A), math.conj(b), math.astensor(complex(1))),
            (A, b, math.astensor(complex(1))),
            range(len(modes)),
            range(len(modes)),
        )
        rho = cls.from_bargmann(list(modes), (A, b, c))
        rho = rho.normalize()
        return rho

    def auto_shape(
        self, max_prob=None, max_shape=None, respect_manual_shape=True
    ) -> tuple[int, ...]:  # TODO: revisit
        r"""
        A good enough estimate of the Fock shape of this DM, defined as the shape of the Fock
        array (batch excluded) if it exists, and if it doesn't exist it is computed as the shape
        that captures at least ``settings.AUTOSHAPE_PROBABILITY`` of the probability mass of each
        single-mode marginal (default 99.9%).
        If the ``respect_manual_shape`` flag is set to ``True``, auto_shape will respect the
        non-None values in ``manual_shape``.

        Args:
            max_prob: The maximum probability mass to capture in the shape (default in ``settings.AUTOSHAPE_PROBABILITY``).
            max_shape: The maximum shape to compute (default in ``settings.AUTOSHAPE_MAX``).
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``.
        """
        # experimental:
        if self.ansatz.batch_size <= 1:
            try:  # fock
                shape = self.ansatz.core_shape
            except AttributeError:  # bargmann
                if self.ansatz.num_derived_vars == 0:
                    ansatz = self.ansatz
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
                    shape = tuple(shape) + tuple(shape)
                else:
                    shape = [settings.AUTOSHAPE_MAX] * 2 * len(self.modes)
        else:
            warnings.warn("auto_shape only looks at the shape of the first element of the batch.")
            shape = [settings.AUTOSHAPE_MAX] * 2 * len(self.modes)
        if respect_manual_shape:
            return tuple(c or s for c, s in zip(self.manual_shape, shape))
        return tuple(shape)

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``DM``.
        """
        return self

    def expectation(self, operator: CircuitComponent):
        r"""
        The expectation value of an operator with respect to this DM.

        Given the operator `O`, this function returns :math:`Tr\big(\rho O)`\, where :math:`\rho`
        is the density matrix of this state.

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

        op_type, msg = _validate_operator(operator)
        if op_type is OperatorType.INVALID_TYPE:
            raise ValueError(msg)

        if not operator.wires.modes.issubset(self.wires.modes):
            msg = f"Expected an operator defined on a subset of modes `{self.modes}`, "
            msg += f"found one defined on `{operator.modes}.`"
            raise ValueError(msg)

        leftover_modes = self.wires.modes - operator.wires.modes
        if op_type is OperatorType.KET_LIKE:
            result = self >> operator.dual
            if leftover_modes:
                result >>= TraceOut(leftover_modes)
        elif op_type is OperatorType.DM_LIKE:
            result = self >> operator.dual
            if leftover_modes:
                result >>= TraceOut(leftover_modes)
        else:
            result = (self.contract(operator)) >> TraceOut(self.modes)

        return result

    def fock_array(
        self, shape: int | Sequence[int] | None = None, standard_order: bool = False
    ) -> ComplexTensor:
        r"""
        Returns an array representation of this component in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        The ``standard_order`` boolean argument lets one choose the standard convention for the
        index ordering of the density matrix. For a single mode, if ``standard_order=True`` the
        returned 2D array :math:`rho_{ij}` has a first index corresponding to the "left" (ket)
        side of the matrix and the second index to the "right" (bra) side. Otherwise, MrMustard's
        convention is that the bra index comes before the ket index. In other words, for a single
        mode, the array returned by ``fock_array`` with ``standard_order=False`` (false by default)
        is the transpose of the standard density matrix. For multiple modes, the same applies to
        each pair of indices of each mode.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is estimated.
            standard_order: The ordering of the wires. If ``standard_order = False``, then the conventional ordering
            of bra-ket is chosen. However, if one wants to get the actual matrix representation in the
            standard conventions of linear algebra, then ``standard_order=True`` must be chosen.
        Returns:
            array: The Fock representation of this component.
        """
        array = super().fock_array(shape or self.auto_shape())
        if standard_order:
            m = self.n_modes
            axes = tuple(range(m, 2 * m)) + tuple(
                range(m)
            )  # to take care of multi-mode case, otherwise, for a single mode we could just use a simple transpose method
            array = math.transpose(array, perm=axes)
        return array

    def normalize(self) -> DM:
        r"""
        Returns a rescaled version of the state such that its probability is 1.
        """
        return self / self.probability

    def formal_stellar_decomposition(self, core_modes: Collection[int]):
        r"""
        Computes the formal stellar decomposition for the DM.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            core: The core state (`DM`)
            phi: The Gaussian `Map` performing the stellar decomposition (not necessarily CPTP).

        Note:
            This method pulls out the map ``phi`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \rho = (\phi\otimes\mathcal I) \rho_{\mathrm{core}}

            where the map :math:`phi` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.

        .. code-block::
            >>> from mrmustard.lab_dev import DM, Vacuum

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.formal_stellar_decomposition([0])
            >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0).dm()
            >>> assert rho == core >> phi
            >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0).dm()
        """

        other_modes = [m for m in self.modes if m not in core_modes]
        core_indices = self.wires[core_modes].indices
        other_indices = self.wires[other_modes].indices
        new_order = core_indices + other_indices
        A, b, c = self.ansatz.reorder(new_order).triple

        M = len(core_modes)

        Am = A[: 2 * M, : 2 * M]
        An = A[2 * M :, 2 * M :]
        R = A[: 2 * M, 2 * M :]
        bm = b[: 2 * M]
        bn = b[2 * M :]

        A_core = math.block([[math.zeros((2 * M, 2 * M), dtype=math.complex128), R], [R.T, An]])
        b_core = math.block([math.zeros(2 * M, dtype=math.complex128), bn], axes=(0, 0))
        c_core = c

        inverse_order = np.argsort(new_order)

        temp = math.astensor(inverse_order)
        A_core = A_core[temp, :]
        A_core = A_core[:, temp]
        b_core = b_core[temp]
        core = DM.from_bargmann(self.modes, (A_core, b_core, c_core))

        I = math.eye_like(Am)
        O = math.zeros_like(Am)
        A_out_in = math.block([[Am, I], [I, O]])
        A_tmp = math.reshape(A_out_in, (2, 2, M, 2, 2, M))
        A_tmp = math.transpose(A_tmp, (1, 0, 2, 4, 3, 5))
        A_T = math.reshape(A_tmp, (4 * M, 4 * M))

        b_out_in = math.block([bm, math.zeros(2 * M, dtype=math.complex128)], axes=(0, 0))
        b_temp = math.reshape(b_out_in, (2, 2, M))
        b_temp = math.transpose(b_temp, (1, 0, 2))
        b_T = math.reshape(b_temp, (4 * M,))
        c_T = math.astensor(1, dtype=math.complex128)
        phi = Map.from_bargmann(core_modes, core_modes, (A_T, b_T, c_T))
        return core, phi

    def physical_stellar_decomposition(self, core_modes: Collection[int]):
        r"""
        Applies the physical stellar decomposition, pulling put a channel from a pure state.

        Args:
            core_modes: the core modes defining the core variables.

        Returns:
            core: The core state (`Ket`)
            phi: The channel acting on the core modes (`Map`)

        Raises:
            ValueError: if the number of core modes is not half the total number of modes.

        Note:
            This method writes a given `DM` as a pure state (`Ket`) followed by a `Channel` acting
            on `core_modes`.
            The pure state output has the core property, and therefore, has favorable Fock representation.
            For the method to work, we need the number of core modes to be half of the number of total modes.

        .. code-block::
            >>> from mrmustard.lab_dev import DM, Ket, Vacuum

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.physical_stellar_decomposition([0])

            >>> assert isinstance(core, Ket)
            >>> assert rho == core >> phi
            >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0)
        """
        from .ket import Ket

        other_modes = [m for m in self.modes if m not in core_modes]
        core_bra_indices = self.wires.bra[core_modes].indices
        core_ket_indices = self.wires.ket[core_modes].indices
        core_indices = core_bra_indices + core_ket_indices

        other_bra_indices = self.wires.bra[other_modes].indices
        other_ket_indices = self.wires.ket[other_modes].indices
        other_indices = other_bra_indices + other_ket_indices

        new_order = core_indices + other_indices
        new_order = math.astensor(core_indices + other_indices)

        A, _, _ = self.ansatz.reorder(new_order).triple

        m_modes = A.shape[-1] // 2

        if (m_modes % 2) or (m_modes // 2 != len(core_modes)):
            raise ValueError(
                f"The number of modes ({m_modes}) must be twice the number of core modes ({len(core_modes)}) for the physical decomposition to work."
            )

        M = len(core_modes)
        Am = A[: 2 * M, : 2 * M]
        An = A[2 * M :, 2 * M :]
        R = A[2 * M :, : 2 * M]

        # computing the core state:
        reduced_A = An - R @ math.inv(Am - math.Xmat(M)) @ R.T
        r_squared = reduced_A[:M, M:]
        r_evals, r_evecs = math.eigh(r_squared)
        r_core = (r_evecs * math.sqrt(r_evals, dtype=math.complex128) @ math.conj(r_evecs.T)).T
        a_core = reduced_A[M:, M:]
        A_core = math.block(
            [
                [math.zeros((M, M), dtype=math.complex128), r_core.T],
                [r_core, a_core],
            ]
        )
        b_core = math.zeros(self.n_modes, dtype=math.complex128)
        c_core = math.astensor(1, dtype=math.complex128)  # to be renormalized

        inverse_order = np.argsort(core_ket_indices + other_ket_indices)
        inverse_order = [i for i in inverse_order if i < self.n_modes]  # removing double-indices
        temp = math.astensor(inverse_order)
        A_core = A_core[temp, :]
        A_core = A_core[:, temp]
        b_core = b_core[temp]
        core = Ket.from_bargmann(self.modes, (A_core, b_core, c_core)).normalize()

        Aphi_out = Am
        Gamma_phi = (
            math.inv(
                math.block(
                    [
                        [
                            math.conj(r_core),
                            math.zeros((M, M), dtype=math.complex128),
                        ],
                        [
                            math.zeros((M, M), dtype=math.complex128),
                            r_core,
                        ],
                    ]
                )
            )
            @ R
        )

        Aphi_in = Gamma_phi @ math.inv(Aphi_out - math.Xmat(M)) @ Gamma_phi.T + math.Xmat(M)

        Aphi_oi = math.block([[Aphi_out, Gamma_phi.T], [Gamma_phi, Aphi_in]])
        A_tmp = math.reshape(Aphi_oi, (2, 2, M, 2, 2, M))
        A_tmp = math.transpose(A_tmp, (1, 0, 2, 4, 3, 5))
        Aphi = math.reshape(A_tmp, (4 * M, 4 * M))

        bphi = math.zeros(4 * M, dtype=math.complex128)
        phi = Channel.from_bargmann(core_modes, core_modes, (Aphi, bphi, 1.0))
        renorm = (core >> phi).probability
        phi = phi / renorm
        return core, phi

    def physical_stellar_decomposition_2(self, core_modes: Collection[int]):
        r"""
        Applies the physical stellar decomposition based on the rank condition.

        Args:
            core_modes: the core modes defining the core variables.

        Returns:
            core: The core state (`DM`)
            phi: The channel acting on the core modes (`Channel`)

        Raises:
            ValueError: if the rank condition is not satisfied.

        .. code-block::

            >>> from mrmustard.lab_dev import DM, Vacuum

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.physical_stellar_decomposition_2([0])

            >>> assert rho == core >> phi
            >>> assert core.is_physical
            >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0).dm()
        """
        other_modes = [m for m in self.modes if m not in core_modes]
        core_bra_indices = self.wires.bra[core_modes].indices
        core_ket_indices = self.wires.ket[core_modes].indices
        core_indices = core_bra_indices + core_ket_indices

        other_bra_indices = self.wires.bra[other_modes].indices
        other_ket_indices = self.wires.ket[other_modes].indices
        other_indices = other_bra_indices + other_ket_indices

        new_order = core_indices + other_indices
        new_order = math.astensor(core_indices + other_indices)

        A, _, _ = self.ansatz.reorder(new_order).triple

        M = len(core_modes)
        N = self.n_modes - M

        Am = A[: 2 * M, : 2 * M]
        An = A[2 * M :, 2 * M :]
        R = A[2 * M :, : 2 * M]

        sigma = R[M:, :M]
        r = R[M:, M:]
        alpha_m = Am[M:, :M]
        alpha_n = An[N:, :N]
        a_n = An[N:, N:]

        rank = np.linalg.matrix_rank(r @ math.conj(r).T + sigma @ math.conj(sigma.T))
        if rank > M:
            raise ValueError(f"The rank {rank} is larger than the number of core modes {M}.")

        reduced_A = R @ math.inv(math.eye(2 * M) - math.Xmat(M) @ Am) @ math.conj(R.T)

        # computing a low-rank r_c:
        r_c_squared = reduced_A[N:, N:] + sigma @ math.inv(alpha_m) @ math.conj(sigma.T)
        r_c_evals, r_c_evecs = math.eigh(r_c_squared)
        idx = np.argsort(r_c_evals)[::-1]
        r_c_evals = r_c_evals[idx]
        r_c_evecs = r_c_evecs[:, idx]
        r_c = r_c_evecs[:, :M] * math.sqrt(r_c_evals[:M], dtype=math.complex128)
        R_c = math.block(
            [
                [math.conj(r_c), math.zeros((N, M), dtype=math.complex128)],
                [math.zeros((M, N), dtype=math.complex128), r_c],
            ]
        )
        alpha_n_c = alpha_n - sigma @ math.inv(alpha_m) @ math.conj(sigma.T)
        a_n_c = a_n + reduced_A[N:, :N]
        An_c = math.block([[math.conj(a_n_c), math.conj(alpha_n_c)], [alpha_n_c, a_n_c]])
        A_core = math.block(
            [[math.zeros((2 * M, 2 * M), dtype=math.complex128), R_c.T], [R_c, An_c]]
        )
        b_core = math.zeros(2 * self.n_modes, dtype=math.complex128)
        c_core = 1

        inverse_order = np.argsort(new_order)

        temp = math.astensor(inverse_order)
        A_core = A_core[temp, :]
        A_core = A_core[:, temp]
        b_core = b_core[temp]
        core = DM.from_bargmann(self.modes, (A_core, b_core, c_core)).normalize()

        Aphi_out = Am
        gamma = np.linalg.pinv(R_c) @ R
        Aphi_in = gamma @ math.inv(Aphi_out - math.Xmat(M)) @ gamma.T + math.Xmat(M)

        Aphi_oi = math.block([[Aphi_out, gamma.T], [gamma, Aphi_in]])
        A_tmp = math.reshape(Aphi_oi, (2, 2, M, 2, 2, M))
        A_tmp = math.transpose(A_tmp, (1, 0, 2, 4, 3, 5))
        Aphi = math.reshape(A_tmp, (4 * M, 4 * M))
        bphi = math.zeros(4 * M, dtype=math.complex128)
        phi = Channel.from_bargmann(core_modes, core_modes, (Aphi, bphi, 1.0))
        renorm = (core >> phi).probability
        phi = phi / renorm

        return core, phi

    def _ipython_display_(self):  # pragma: no cover
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        is_fock = isinstance(self.ansatz, ArrayAnsatz)
        display(widgets.state(self, is_ket=False, is_fock=is_fock))

    def __getitem__(self, idx: int | Sequence[int]) -> State:
        r"""
        Traces out all the modes except those given.
        The result is returned with modes in increasing order.

        Args:
            idx: The modes to keep.

        Returns:
            A new DM with the modes indexed by `idx`.
        """
        idx = (idx,) if isinstance(idx, int) else idx
        modes = set(idx)
        if not modes.issubset(self.modes):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{idx}``.")
        wires = Wires(modes_out_bra=modes, modes_out_ket=modes)
        idxz = [i for i, m in enumerate(self.modes) if m not in modes]
        idxz_conj = [i + len(self.modes) for i, m in enumerate(self.modes) if m not in modes]
        ansatz = self.ansatz.trace(idxz, idxz_conj)
        return DM(Representation(ansatz, wires), self.name)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` (output of self into the inputs of other),
        adding the adjoints when they are missing. Given this is a ``DM`` object which
        has both ket and bra wires at the output, expressions like ``dm >> u`` where
        ``u`` is a unitary will automatically apply the adjoint of ``u`` on the bra side.

        Returns a ``DM`` when the wires of the resulting components are compatible with
        those of a ``DM``, a ``CircuitComponent`` otherwise, and a scalar if there are no wires left.
        """

        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input and result.wires.bra.modes == result.wires.ket.modes:
            return DM(result.representation)
        return result
