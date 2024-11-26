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
from typing import Sequence

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
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexMatrix, ComplexVector, ComplexTensor, RealVector

from .base import State, _validate_operator, OperatorType
from ..circuit_components import CircuitComponent
from ..circuit_components_utils import BtoQ, TraceOut

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
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
    ) -> State:
        return DM.from_ansatz(modes, PolyExpAnsatz(*triple), name)

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batched: bool = False,
    ) -> State:
        return DM.from_ansatz(modes, ArrayAnsatz(array, batched), name)

    @classmethod
    def from_ansatz(
        cls,
        modes: Sequence[int],
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
            wires = wires.to_fock(ansatz.array.shape)
        return DM(Representation(ansatz, wires), name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
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
    def from_quadrature(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        phi: float = 0.0,
        name: str | None = None,
    ) -> State:
        r"""
        Initializes a state from a triple (A,b,c) that parametrizes the wavefunction
        as `c * exp(0.5 z^T A z + b^T z)` in the quadrature representation.

        Args:
            modes: The modes of this state.
            triple: The ``(A, b, c)`` triple.
            phi: The angle of the quadrature. 0 corresponds to the x quadrature (default).
            name: The name of this state.

        Returns:
            A state of type ``cls``.

        Raises:
            ValueError: If the given triple has shapes that are inconsistent
                with the number of modes.
        """

        QtoB = BtoQ(modes, phi).inverse()
        Q = DM.from_ansatz(modes, PolyExpAnsatz(*triple))
        return DM.from_ansatz(modes, (Q >> QtoB).ansatz, name)

    @classmethod
    def random(cls, modes: Sequence[int], m: int | None = None, max_r: float = 1.0) -> DM:
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
                if self.ansatz.num_DV_vars == 0:
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

    def _ipython_display_(self):  # pragma: no cover
        is_fock = isinstance(self.ansatz, ArrayAnsatz)
        display(widgets.state(self, is_ket=False, is_fock=is_fock))

    def __getitem__(self, modes: int | Sequence[int]) -> State:
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

        if self._parameter_set:
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
