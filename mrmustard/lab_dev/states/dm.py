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

from itertools import product
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
    def is_positive(self) -> bool:
        r"""
        Whether this DM is a positive operator.
        """
        batch_dim = self.ansatz.batch_size
        if batch_dim > 1:
            raise ValueError(
                "Physicality conditions are not implemented for batch dimension larger than 1."
            )
        A = self.ansatz.A[0]
        m = A.shape[-1] // 2
        gamma_A = A[:m, m:]

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
        Samples a random density matrix. The final state has zero displacement.

        Args:
        modes: the modes where the state is defined over
        m: is the number modes to be considered for tracing out from a random pure state (Ket)
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
            (math.conj(A), math.conj(b), complex(1)),
            (A, b, complex(1)),
            range(len(modes)),
            range(len(modes)),
        )
        rho = cls.from_bargmann(list(modes), (A, b, c))
        rho = rho.normalize()
        return rho

    def auto_shape(
        self, max_prob=None, max_shape=None, respect_manual_shape=True
    ) -> tuple[int, ...]:
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
        if self.ansatz.batch_size == 1:
            try:  # fock
                shape = self.ansatz.array.shape[1:]
            except AttributeError:  # bargmann
                if self.ansatz.polynomial_shape[0] == 0:
                    ansatz = self.ansatz
                    A, b, c = ansatz.A[0], ansatz.b[0], ansatz.c[0]
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
            result = (self @ operator) >> TraceOut(self.modes)

        return result

    def fock_array(
        self, shape: int | Sequence[int] | None = None, batched=False, standard_order: bool = False
    ) -> ComplexTensor:
        r"""
                Returns an array representation of this component in the Fock basis with the given shape.
                If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
                available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        The ``standard_order`` boolean argument lets one choose the standard convention for the index ordering of the density matrix. For a single mode, if ``standard_order=True`` the returned 2D array :math:`rho_{ij}` has a first index corresponding to the "left" (ket) side of the matrix and the second index to the "right" (bra) side. Otherwise, MrMustard's convention is that the bra index comes before the ket index. In other words, for a single mode, the array returned by ``fock_array`` with ``standard_order=False`` (false by default) is the transpose of the standard density matrix. For multiple modes, the same applies to each pair of indices of each mode.

                Args:
                    shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                        it is broadcasted to all the dimensions. If not given, it is estimated.
                    batched: Whether the returned representation is batched or not. If ``False`` (default)
                        it will squeeze the batch dimension if it is 1.
                    standard_order: The ordering of the wires. If ``standard_order = False``, then the conventional ordering
                    of bra-ket is chosen. However, if one wants to get the actual matrix representation in the
                    standard conventions of linear algebra, then ``standard_order=True`` must be chosen.
                Returns:
                    array: The Fock representation of this component.
        """
        array = super().fock_array(shape or self.auto_shape(), batched)
        if standard_order:
            m = self.n_modes
            axes = tuple(range(m, 2 * m)) + tuple(
                range(m)
            )  # to take care of multi-mode case, otherwise, for a single mode we could just use a simple transpose method
            array = math.transpose(array, perm=axes)
        return array

    def fock_distribution(self, cutoff: int) -> ComplexTensor:
        r"""
        Returns the Fock distribution of the state up to some cutoff.

        Args:
            cutoff: The photon cutoff.

        Returns:
            The Fock distribution.
        """
        fock_array = self.fock_array(cutoff)
        return math.astensor(
            [fock_array[ns * 2] for ns in product(list(range(cutoff)), repeat=self.n_modes)]
        )

    def normalize(self) -> DM:
        r"""
        Returns a rescaled version of the state such that its probability is 1.
        """
        return self / self.probability

    def quadrature_distribution(self, quad: RealVector, phi: float = 0.0) -> ComplexTensor:
        r"""
        The (discretized) quadrature distribution of the State.

        Args:
            quad: the discretized quadrature axis over which the distribution is computed.
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
        Returns:
            The quadrature distribution.
        """
        quad = np.array(quad)
        if len(quad.shape) != 1 and len(quad.shape) != self.n_modes:
            raise ValueError(
                "The dimensionality of quad should be 1, or match the number of modes."
            )

        if len(quad.shape) == 1:
            quad = math.astensor(np.meshgrid(*[quad] * len(self.modes))).T.reshape(
                -1, len(self.modes)
            )

        quad = math.tile(quad, (1, 2))
        return self.quadrature(quad, phi)

    def stellar_decomposition(self, core_modes: Collection[int]):
        r"""
        Computes the formal stellar decomposition for the DM.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            core: The core state (`DM`)
            phi: The Gaussian `Map` performing the stellar decomposition (not necessarily CPTP).

        Raises:
            ValueError: if the state is non-Gaussian.

        Note:
            This method pulls out the map ``phi`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \rho = (\phi\otimes\mathcal I) \rho_{\mathrm{core}}

            where the map :math:`phi` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.

        .. code-block::
            >>> from mrmustard.lab_dev import DM

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.formal
            >>> assert rho == core >> phi
        """
        A, b, c = self.ansatz.triple
        A = A[-1]
        b = b[-1]
        if c.shape != (1,):
            raise ValueError(
                f"The stellar decomposition only applies to Gaussian states. The given state has a polynomial of size {c.shape}."
            )

        m_modes = A.shape[-1] // 2

        mode_to_idx = {q.mode: q.index for q in self.wires.quantum_wires}
        core_bra_indices = [
            (i - m_modes) if i >= m_modes else i for i in (mode_to_idx[j] for j in core_modes)
        ]
        core_ket_indices = [(i + m_modes) for i in core_bra_indices]
        core_indices = core_bra_indices + core_ket_indices
        remaining_indices = [i for i in range(2 * m_modes) if i not in core_indices]
        new_order = math.astensor(core_indices + remaining_indices)

        A_reordered = A[new_order, :]
        A_reordered = A_reordered[
            :, new_order
        ]  # reordering indices of A so that it has the standard form.
        b_reordered = b[new_order]

        core_size = 2 * len(core_modes)
        Am = A_reordered[:core_size, :core_size]
        An = A_reordered[core_size:, core_size:]
        R = A_reordered[:core_size, core_size:]
        bm = b_reordered[:core_size]
        bn = b[core_size:]

        # core state's Abc
        A_core = math.block(
            [[math.zeros((core_size, core_size), dtype=math.complex128), R], [R.T, An]]
        )
        b_core = math.block([math.zeros(core_size, dtype=math.complex128), bn], axes=(0, 0))
        c_core = c

        inverse_order = [orig for orig, _ in sorted(enumerate(new_order), key=lambda x: x[1])]

        temp = math.astensor(inverse_order)
        A_core = A_core[temp, :]
        A_core = A_core[:, temp]
        b_core = b_core[temp]
        core = DM.from_bargmann(self.modes, (A_core, b_core, c_core))

        # the transformation's Abc
        A_T = math.block(
            [
                [
                    Am[: core_size // 2, : core_size // 2],
                    math.eye(core_size // 2, dtype=math.complex128),
                    Am[: core_size // 2, core_size // 2 :],
                    math.zeros((core_size // 2, core_size // 2), dtype=math.complex128),
                ],
                [
                    math.eye(core_size // 2, dtype=math.complex128),
                    math.zeros((core_size // 2, 3 * core_size // 2), dtype=math.complex128),
                ],
                [
                    Am[core_size // 2 :, : core_size // 2],
                    math.zeros((core_size // 2, core_size // 2), dtype=math.complex128),
                    Am[core_size // 2 :, core_size // 2 :],
                    math.eye(core_size // 2, dtype=math.complex128),
                ],
                [
                    math.zeros((core_size // 2, core_size), dtype=math.complex128),
                    math.eye(core_size // 2, dtype=math.complex128),
                    math.zeros((core_size // 2, core_size // 2), dtype=math.complex128),
                ],
            ]
        )

        b_T = math.block(
            [
                bm[: core_size // 2],
                math.zeros(core_size // 2, dtype=math.complex128),
                bm[core_size // 2 :],
                math.zeros(core_size // 2, dtype=math.complex128),
            ],
            axes=(0, 0, 0, 0),
        )
        c_T = 1
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
            ValueError: if the given state is non-Gaussian
            ValueError: if the number of core modes is not half the total number of modes.

        Note:
            This method writes a given `DM` as a pure state (`Ket`) followed by a `Channel` acting
            on `core_modes`.
            The pure state output has the core property, and therefore, has favorable Fock representation.
            For the method to work, we need the number of core modes to be half of the number of total modes.

        .. code-block::
            >>> from mrmustard.lab_dev import DM

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.physical_stellar_decomposition([0])

            >>> assert isinstance(core, Ket)
            >>> assert rho == core >> phi
        """
        from .ket import Ket

        A, b, c = self.ansatz.triple
        A = A[-1]
        b = b[-1]
        if c.shape != (1,):
            raise ValueError(
                f"The stellar decomposition only applies to Gaussian states. The given state has a polynomial of size {c.shape}."
            )

        m_modes = A.shape[-1] // 2

        if (m_modes % 2) or (m_modes // 2 != len(core_modes)):
            raise ValueError(
                f"The number of modes ({m_modes}) must be twice the number of core modes ({len(core_modes)}) for the physical decomposition to work."
            )

        mode_to_idx = {q.mode: q.index for q in self.wires.quantum_wires}
        core_bra_indices = [
            (i - m_modes) if i >= m_modes else i for i in (mode_to_idx[j] for j in core_modes)
        ]
        core_ket_indices = [(i + m_modes) for i in core_bra_indices]
        core_indices = core_bra_indices + core_ket_indices
        remaining_indices = [i for i in range(2 * m_modes) if i not in core_indices]
        new_order = math.astensor(core_indices + remaining_indices)

        A_reordered = A[new_order, :]
        A_reordered = A_reordered[
            :, new_order
        ]  # reordering indices of A so that it has the standard form.

        num_core_modes = len(core_modes)
        core_size = 2 * num_core_modes

        Am = A_reordered[:core_size, :core_size]
        An = A_reordered[core_size:, core_size:]
        R = A_reordered[:core_size, core_size:]

        # computing the core state:
        reduced_A = An - R.T @ math.inv(Am - math.Xmat(num_core_modes)) @ R
        r_squared = reduced_A[:num_core_modes, num_core_modes:]
        r_evals, r_evecs = math.eigh(r_squared)
        r_core = (r_evecs * math.sqrt(r_evals) @ math.conj(r_evecs.T)).T
        a_core = reduced_A[num_core_modes:, num_core_modes:]
        A_core = math.block(
            [
                [math.zeros((num_core_modes, num_core_modes), dtype=math.complex128), r_core.T],
                [r_core, a_core],
            ]
        )
        b_core = math.zeros(self.n_modes, dtype=math.complex128)
        c_core = 1  # to be renormalized

        core = Ket.from_bargmann(self.modes, (A_core, b_core, c_core)).normalize()

        Aphi_out = Am
        Gamma_phi = (
            math.inv(
                math.block(
                    [
                        [
                            math.conj(r_core),
                            math.zeros((num_core_modes, num_core_modes), dtype=math.complex128),
                        ],
                        [
                            math.zeros((num_core_modes, num_core_modes), dtype=math.complex128),
                            r_core,
                        ],
                    ]
                )
            )
            @ R
        )

        Aphi_in = Gamma_phi.T @ math.inv(
            Aphi_out - math.Xmat(num_core_modes)
        ) @ Gamma_phi + math.Xmat(num_core_modes)

        Aphi = math.block([[Aphi_out, Gamma_phi], [Gamma_phi, Aphi_in]])
        type_wise_order = (
            list(range(num_core_modes))
            + list(range(2 * num_core_modes, 3 * num_core_modes))
            + list(range(num_core_modes, 2 * num_core_modes))
            + list(range(3 * num_core_modes, 4 * num_core_modes))
        )

        Aphi = Aphi[type_wise_order, :]
        Aphi = Aphi[:, type_wise_order]
        bphi = math.zeros(4 * num_core_modes, dtype=math.complex128)
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

    def __getitem__(self, modes: int | Collection[int]) -> State:
        r"""
        Traces out all the modes except those given.
        The result is returned with modes in increasing order.
        """
        if isinstance(modes, int):
            modes = {modes}
        modes = set(modes)

        if not modes.issubset(self.modes):
            msg = f"Expected a subset of `{self.modes}, found `{list(modes)}`."
            raise ValueError(msg)

        if self._parameters:
            # if ``self`` has a parameter set it means it is a built-in state,
            # in which case we slice the parameters
            return self._getitem_builtin(modes)

        # if ``self`` has no parameter set it is not a built-in state,
        # in which case we trace the representation
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
