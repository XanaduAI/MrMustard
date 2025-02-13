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
from mrmustard.utils.typing import ComplexTensor, RealVector, ComplexMatrix, ComplexVector

from .base import State, _validate_operator, OperatorType
from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut

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
        Whether this DM corresponds to a positive operator.

        Raises:
            ValueError: if the batch dimension of the state is greater than 1.

        Example:
            .. code-block::
                >>> from mrmustard.lab_dev import DM
                >>> assert DM.random([0]).is_positive
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

        Raises:
            ValueError: if the batch dimension of the state is greater than 1.

        Example:
            .. code-block::
                >>> from mrmustard.lab_dev import DM
                >>> assert DM.random([0]).is_physical
        """
        return self.is_positive and math.allclose(self.probability, 1, settings.ATOL)

    @property
    def probability(self) -> float:
        r"""
        Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states.

        Example:
        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import DM
            >>> assert np.isclose(DM.random([0]).probability, 1.0)
        """
        return math.sum(self._probabilities)

    @property
    def purity(self) -> float:
        r"""Computes the putiry (:math:`tr(rho^2)) of this DM.

        Example:
            >>> import numpy as np
            >>> from mrmustard.lab_dev import DM, Vacuum
            >>> assert np.isclose(Vacuum([0]).dm().purity, 1.0)
        """
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
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
        atol_purity: None = None,  # pylint: disable=unused-argument
    ) -> DM:
        r"""
        Initializes a density matrix from the covariance matrix, vector of means and a coefficient,
        which parametrize the Wigner function.

        Args:
            modes: The modes of this states.
            triple: The ``(cov, means, coeff)`` triple.
            name: The name of this state.
            atol_purity: Should be set to None.

        Returns:
            A ``DM`` object from its phase space representation.

        .. details::
            The Wigner function is considered as
            :math:`coeff * exp(-1/2(x-means)^T cov^{-1} (x-means))`.h:`coeff * exp((x-means)^T cov^{-1} (x-means))`.

        Example:
        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import DM, Vacuum
            >>> rho = DM.from_phase_space([0], (np.eye(2)/2, [0,0], 1))
            >>> assert rho == Vacuum([0]).dm()
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
        Samples a random density matrix with no displacement.

        Args:
            modes: the modes where the state is defined over
            m: is the number modes to be considered for tracing out from a random pure state (Ket)
            if not specified, m is considered to be len(modes)
            max_r: maximum squeezing chosen for the generation of the random state.

        Returns:
            A ``DM``.

        .. detail::
            Using a random Gaussian unitary, :math:`U`, on :math:`len(modes)+m`, the code outputs
            :math:`\mathrm{tr}_{m}(U|0\rangle)`. The random unitary :math:`U` is chosen with maximum
            squeezing determined by `max_r`.

        Example:
        .. code-block::
            >>> from mrmustard.lab_dev import DM
            >>> assert isinstance(DM.random([0,1]), DM)
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
            max_prob: The maximum probability mass to capture in the shape (default in
            ``settings.AUTOSHAPE_PROBABILITY``).
            max_shape: The maximum shape to compute (default in ``settings.AUTOSHAPE_MAX``).
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``.

        Returns:
            A ``tuple`` demonstrating the Fock cutoffs along each axis.

        Raises:
            Warning: if the item is batched. In that case, the `auto_shape` will only output the
            shape computed for the first element in the batch.

        Example:
        .. code-block::
            >>> from mrmustard.lab_dev import Vacuum
            >>> assert Vacuum([0]).dm().auto_shape() == (1,1)
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

        Returns:
            A ``DM``.

        Example:
        .. code-block:
            >>> from mrmustard.lab_dev import Vacuum, DM
            >>> assert isinstance(Vacuum([0]).dm(), DM)
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

        Returns:
            Expectation value as a float.

        Raise:
            ValueError: If ``operator`` is not a ket-like, density-matrix like, or unitary-like
                component.
            ValueError: If ``operator`` is defined over a set of modes that is not a subset of the
                modes of this state.

        Example:
            In the example below, we consider a thermal state and compute its expectation value against
            the parity operator, which is equivalent to `Rgate([0], np.pi)`.

        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import Rgate, GDM
            >>> beta = 1
            >>> symplectic = np.eye(2)
            >>> rho = GDM([0], beta, symplectic)
            >>> answer = (1-np.exp(-beta))/(1+np.exp(-beta))
            >>> assert np.isclose(rho.expectation(Rgate([0], np.pi)), answer)
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

        Note:
            The ``standard_order`` boolean argument lets one choose the standard convention for the
            index ordering of the density matrix. For a single mode, if ``standard_order=True`` the
            returned 2D array :math:`rho_{ij}` has a first index corresponding to the "left" (ket)
            side of the matrix and the second index to the "right" (bra) side. Otherwise, MrMustard's
            convention is that the bra index comes before the ket index. In other words, for a single
            mode, the array returned by ``fock_array`` with ``standard_order=False`` (false by default)
            is the transpose of the standard density matrix. For multiple modes, the same applies to each
            pair of indices of each mode.

        Example:
        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import Vacuum, DM
            >>> assert np.allclose(Vacuum([0]).dm().fock_array(), np.array([[1]]))
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

        Example:
        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import Vacuum, DM
            >>> assert np.allclose(Vacuum([0]).dm().fock_distribution(2), np.array([1, 0]))
        """
        fock_array = self.fock_array(cutoff)
        return math.astensor(
            [fock_array[ns * 2] for ns in product(list(range(cutoff)), repeat=self.n_modes)]
        )

    def normalize(self) -> DM:
        r"""
        Normalizes the state so that it has unit trace.

        Returns:
            A ``DM``.

        Example:
        .. code-block::
            >>> import numpy as np
            >>> from mrmustard.lab_dev import DM
            >>> rho = DM.random([0,1]) * 2
            >>> assert np.isclose(rho.probability, 2.0)
            >>> assert np.isclose(rho.normalize().probability, 1.0)
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

        Example:
        .. code-block::
            >>> from mrmustard.lab_dev import DM
            >>> dist = DM.random([0]).dm().quadrature_distribution(np.linspace(-2,2,20))
            >>> assert all((dist.real) >= 0)
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
        adding the adjoints when they are missing.

        Args:
            other: the ``CircuitComponent`` we want to contract with.

        Returns:
            A ``DM`` when the wires of the resulting components are compatible with
            those of a ``DM``, a ``CircuitComponent`` otherwise, and a scalar if there are no wires left.

        Note:
            Given this is a ``DM`` object which
            has both ket and bra wires at the output, expressions like ``dm >> u`` where
            ``u`` is a unitary will automatically apply the adjoint of ``u`` on the bra side.

        Example:
        .. code-block::
            >>> from mrmustard.lab_dev import CircuitComponent, DM, TraceOut
            >>> assert isinstance(DM.random([0]).dual >> DM.random([0]), CircuitComponent)
            >>> assert isinstance(DM.random([0,1]) >> TraceOut([0]), DM)
        """

        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input and result.wires.bra.modes == result.wires.ket.modes:
            return DM(result.representation)
        return result
