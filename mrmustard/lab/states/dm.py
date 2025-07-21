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

from collections.abc import Collection, Sequence

import numpy as np
from IPython.display import display

from mrmustard import math, settings, widgets
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_rho
from mrmustard.physics.fock_utils import fidelity as fock_dm_fidelity
from mrmustard.physics.gaussian import fidelity as gaussian_fidelity
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from mrmustard.physics.utils import outer_product_batch_str
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Scalar

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..transformations import Channel, Dgate, Map
from ..utils import shape_check
from .base import OperatorType, State, _validate_operator

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
            NotImplementedError: If the state is mixed.
            NotImplementedError: If the state has derived variables.
            NotImplementedError: If the state has an ``ArrayAnsatz``.

        .. code-block::

            >>> from mrmustard.lab import DM
            >>> assert DM.random([0]).is_positive
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError(
                "Physicality conditions are not implemented for a mixture of states.",
            )
        if self.ansatz.num_derived_vars > 0:
            raise ValueError("Physicality conditions are not implemented for derived variables.")
        if isinstance(self.ansatz, ArrayAnsatz):
            raise NotImplementedError(
                "Physicality conditions are not implemented for states with ArrayAnsatz.",
            )
        A = self.ansatz.A
        m = A.shape[-1] // 2
        gamma_A = A[..., :m, m:]

        if (
            math.real(math.norm(gamma_A - math.conj(math.einsum("...ij->...ji", gamma_A))))
            > settings.ATOL
        ):  # checks if gamma_A is Hermitian
            return False

        return math.all(math.real(math.eigvals(gamma_A)) >= -settings.ATOL)

    @property
    def is_physical(self) -> bool:
        r"""
        Whether this DM is a physical density operator.

        .. code-block::

            >>> from mrmustard.lab import DM
            >>> assert DM.random([0]).is_physical
        """
        return self.is_positive and math.allclose(self.probability, 1, settings.ATOL)

    @property
    def probability(self) -> float:
        r"""
        Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states.
        """
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices
        rep = self.ansatz.trace(idx_ket, idx_bra)
        return math.real(rep.scalar)

    @property
    def purity(self) -> float:
        r"""
        Computes the purity (:math:`tr(rho^2)`) of this DM.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import DM, Vacuum

            >>> assert math.allclose(Vacuum([0]).dm().purity, 1.0)
        """
        return self.L2_norm / self.probability**2

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
        if ansatz and ansatz.core_dims != 2 * len(modes):
            raise ValueError(
                f"Expected an ansatz with {2 * len(modes)} variables, found {ansatz.core_dims}.",
            )
        wires = Wires(modes_out_bra=set(modes), modes_out_ket=set(modes))
        return DM(ansatz, wires, name=name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
        atol_purity: float | None = None,
    ) -> DM:
        r"""
        Initializes a density matrix from the covariance matrix, vector of means and a coefficient,
        which parametrize the s-parametrized phase space function
        :math:`coeff * exp(-1/2(x-means)^T cov^{-1} (x-means))`.h:`coeff * exp((x-means)^T cov^{-1} (x-means))`.

        Args:
            modes: The modes of this states.
            triple: The ``(cov, means, coeff)`` triple.
            name: The name of this state.
            atol_purity: Unused argument.

        Returns:
            A ``DM`` object from its phase space representation.


        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import DM, Vacuum

            >>> rho = DM.from_phase_space([0], (math.eye(2)/2, [0,0], 1))

            >>> assert rho == Vacuum([0]).dm()

        .. details::

            The Wigner function is considered as
            :math:`coeff * exp(-1/2(x-means)^T cov^{-1} (x-means))`.h:`coeff * exp((x-means)^T cov^{-1} (x-means))`.
        """
        cov, means, coeff = triple
        cov = math.astensor(cov)
        means = math.astensor(means)
        if cov.shape[:-2] != ():
            raise NotImplementedError("Not implemented for batched states.")
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
        return rho.normalize()

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``DM``.

        Returns:
            A ``DM``.

        .. code-block::

            >>> from mrmustard.lab import Vacuum, DM
            >>> assert isinstance(Vacuum([0]).dm(), DM)
        """
        return self

    def expectation(self, operator: CircuitComponent, mode: str = "kron") -> Batch[Scalar]:
        r"""
        The expectation value of an operator with respect to this DM.

        Given the operator `O`, this function returns :math:`Tr\big(\rho O)`\, where :math:`\rho`
        is the density matrix of this state.

        The ``operator`` is expected to be a component with ket-like wires (i.e., output wires on
        the ket side), density matrix-like wires (output wires on both ket and bra sides), or
        unitary-like wires (input and output wires on the ket side).

        Args:
            operator: A ket-like, density-matrix like, or unitary-like circuit component.
            mode: The mode of contraction. Can either "zip" the batch dimensions, "kron" the batch dimensions,
                or pass a custom einsum-style batch string like "ab,cb->ac".
        Returns:
            Expectation value either as a complex number or a batch of complex numbers.

        Raise:
            ValueError: If ``operator`` is not a ket-like, density-matrix like, or unitary-like
                component.
            ValueError: If ``operator`` is defined over a set of modes that is not a subset of the
                modes of this state.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Rgate, GDM

            >>> beta = 1
            >>> symplectic = math.eye(2)
            >>> rho = GDM([0], beta, symplectic)
            >>> answer = (1-math.exp(-beta))/(1+math.exp(-beta))

            >>> assert math.allclose(rho.expectation(Rgate(0, np.pi)), answer)
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
            # if mode is not zip we need to generate a new eins_str for the second contraction
            if mode != "zip":
                eins_str = (
                    outer_product_batch_str(
                        self.ansatz.batch_dims - self.ansatz._lin_sup,
                        operator.ansatz.batch_dims - operator.ansatz._lin_sup,
                    )
                    if mode == "kron"
                    else mode
                )
                batch_in, batch_out = eins_str.split("->")
                _, batch2 = batch_in.split(",")
                eins_str2 = f"{batch_out},{batch2}->{batch_out}"
            else:
                eins_str = mode
                eins_str2 = mode
            result = self.contract(operator.dual.adjoint, mode=eins_str).contract(
                operator.dual,
                mode=eins_str2,
            ) >> TraceOut(leftover_modes)
        elif op_type is OperatorType.DM_LIKE:
            result = self.contract(operator.dual, mode=mode) >> TraceOut(leftover_modes)
        else:
            result = (self.contract(operator, mode=mode)) >> TraceOut(self.modes)

        return result

    def fidelity(self, other: State) -> float:
        r"""
        The fidelity between this DM and another ket or DM. If the other state is a Ket, fidelity
        is computed as the squared overlap, consistent with the pure state's fidelity.
        If the other state is a DM and the representation is Fock, the fidelity is computed as in
        Richard Jozsa (1994) Fidelity for Mixed Quantum States,
        Journal of Modern Optics, 41:12, 2315-2323, DOI: 10.1080/09500349414552171
        Otherwise, the fidelity is computed as the Gaussian fidelity as in
        arXiv:2102.05748 <https://arxiv.org/pdf/2102.05748.pdf> (square definition).

        Args:
            other: The other state.

        Returns:
            The fidelity between this DM and the other state (Ket or DM).

        Raises:
            NotImplementedError: If the state is batched.
            ValueError: If the states have different modes.
        """
        if self.ansatz.batch_shape or other.ansatz.batch_shape:
            raise NotImplementedError("Batched fidelity is not implemented.")
        if self.modes != other.modes:
            raise ValueError("Cannot compute fidelity between states with different modes.")
        if isinstance(other, DM):
            try:
                cov1, mean1, _ = self.phase_space(0)
                cov2, mean2, _ = other.phase_space(0)
                return gaussian_fidelity(mean1, cov1, mean2, cov2)
            except ValueError:  # array ansatz
                shape1 = self.auto_shape()
                shape2 = other.auto_shape()
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(shape1, shape2))
                slc = tuple(slice(None, s) for s in min_shape)
                side = np.prod([min_shape[i] for i in range(len(min_shape) // 2)])
                dm1 = math.reshape(self.fock_array(min_shape)[slc], (side, side))
                dm2 = math.reshape(other.fock_array(min_shape)[slc], (side, side))
                return fock_dm_fidelity(dm1, dm2)
        return other.expectation(self)  # assuming other is a ket

    def fock_array(
        self,
        shape: int | Sequence[int] | None = None,
        standard_order: bool = False,
    ) -> ComplexTensor:
        r"""
        Returns an array representation of this component in the Fock basis with the given shape.

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
                it is broadcasted to all the dimensions. If not given, it is generated via ``auto_shape``.

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

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Vacuum, DM

            >>> assert math.allclose(Vacuum([0]).dm().fock_array(), math.astensor([[1]]))
        """
        array = super().fock_array(shape or self.auto_shape())
        if standard_order:
            m = self.n_modes
            batch_dims = self.ansatz.batch_dims - self.ansatz._lin_sup
            axes = (
                tuple(range(batch_dims))
                + tuple(range(batch_dims + m, 2 * m + batch_dims))
                + tuple(range(batch_dims, batch_dims + m))
            )  # to take care of multi-mode case, otherwise, for a single mode we could just use a simple transpose method
            array = math.transpose(array, perm=axes)

        return array

    def formal_stellar_decomposition(self, core_modes):
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

            >>> from mrmustard.lab import DM, Vacuum

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
        batch_shape = self.ansatz.batch_shape
        Am = A[..., : 2 * M, : 2 * M]
        An = A[..., 2 * M :, 2 * M :]
        R = A[..., : 2 * M, 2 * M :]
        bm = b[..., : 2 * M]
        bn = b[..., 2 * M :]
        R_transpose = math.einsum("...ij->...ji", R)

        A_core = math.block(
            [
                [math.zeros((*batch_shape, 2 * M, 2 * M), dtype=math.complex128), R],
                [R_transpose, An],
            ],
        )
        b_core = math.concat([math.zeros((*batch_shape, 2 * M), dtype=math.complex128), bn], -1)
        c_core = c

        inverse_order = np.argsort(new_order)

        temp = math.astensor(inverse_order)
        A_core = A_core[..., temp, :]
        A_core = A_core[..., :, temp]
        b_core = b_core[..., temp]
        core = DM.from_bargmann(self.modes, (A_core, b_core, c_core))

        I = math.broadcast_to(math.eye(2 * M, dtype=math.complex128), (*batch_shape, 2 * M, 2 * M))
        O = math.zeros_like(Am)
        A_out_in = math.block([[Am, I], [I, O]])
        A_tmp = math.reshape(A_out_in, (*batch_shape, 2, 2, M, 2, 2, M))
        A_tmp = math.einsum("...ijklmn->...jikmln", A_tmp)
        A_T = math.reshape(A_tmp, (*batch_shape, 4 * M, 4 * M))

        b_out_in = math.concat([bm, math.zeros((*batch_shape, 2 * M), dtype=math.complex128)], -1)
        b_temp = math.reshape(b_out_in, (*batch_shape, 2, 2, M))
        b_temp = math.einsum("...ijk->...jik", b_temp)
        b_T = math.reshape(b_temp, (*batch_shape, 4 * M))
        c_T = math.ones_like(c)
        phi = Map.from_bargmann(core_modes, core_modes, (A_T, b_T, c_T))
        return core, phi

    def physical_stellar_decomposition(self, core_modes):
        r"""
        Applies the physical stellar decomposition, pulling out a channel from a pure state.

        Args:
            core_modes: the core modes defining the core variables.

        Returns:
            core: The core state (`Ket`)
            phi: The channel acting on the core modes (`Map`)

        Raises:
            ValueError: If the number of core modes is not half the total number of modes.

        Note:
            This method writes a given `DM` as a pure state (`Ket`) followed by a `Channel` acting
            on `core_modes`.
            The pure state output has the core property, and therefore, has favorable Fock representation.
            For the method to work, we need the number of core modes to be half of the number of total modes.

        .. code-block::

            >>> from mrmustard.lab import DM, Ket, Vacuum
            >>> rho = DM.random([0,1])
            >>> core, phi = rho.physical_stellar_decomposition([0])

            >>> assert isinstance(core, Ket)
            >>> assert rho == core >> phi
            >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0)
        """
        from .ket import Ket  # noqa: PLC0415

        other_modes = [m for m in self.modes if m not in core_modes]
        core_bra_indices = self.wires.bra[core_modes].indices
        core_ket_indices = self.wires.ket[core_modes].indices
        core_indices = core_bra_indices + core_ket_indices

        other_bra_indices = self.wires.bra[other_modes].indices
        other_ket_indices = self.wires.ket[other_modes].indices
        other_indices = other_bra_indices + other_ket_indices

        new_order = math.astensor(core_indices + other_indices)

        batch_shape = self.ansatz.batch_shape
        A, b, c = self.ansatz.reorder(new_order).triple

        m_modes = A.shape[-1] // 2

        if (m_modes % 2) or (m_modes // 2 != len(core_modes)):
            raise ValueError(
                f"The number of modes ({m_modes}) must be twice the number of core modes ({len(core_modes)}) for the physical decomposition to work.",
            )

        M = len(core_modes)
        Am = A[..., : 2 * M, : 2 * M]
        An = A[..., 2 * M :, 2 * M :]
        bm = b[..., : 2 * M]
        bn = b[..., 2 * M :]
        R = A[..., 2 * M :, : 2 * M]
        R_transpose = math.einsum("...ij->...ji", R)
        # computing the core state:
        reduced_A = An - R @ math.inv(Am - math.Xmat(M)) @ R_transpose
        r_squared = reduced_A[..., :M, M:]
        r_evals, r_evecs = math.eigh(r_squared)

        r_core_transpose = math.einsum(
            "...ij,...j,...kj->...ik",
            r_evecs,
            math.sqrt(r_evals),
            math.conj(r_evecs),
        )
        r_core = math.einsum("...ij->...ji", r_core_transpose)

        Aphi_out = Am
        Os = math.zeros(batch_shape + (M,) * 2, dtype=math.complex128)
        temp = math.block([[math.conj(r_core), Os], [Os, r_core]])
        Gamma_phi = math.inv(temp) @ R

        Gamma_phi_transpose = math.einsum("...ij->...ji", Gamma_phi)
        Aphi_in = Gamma_phi @ math.inv(Aphi_out - math.Xmat(M)) @ Gamma_phi_transpose + math.Xmat(M)

        Aphi_oi = math.block([[Aphi_out, Gamma_phi_transpose], [Gamma_phi, Aphi_in]])
        A_tmp = math.reshape(Aphi_oi, (*batch_shape, 2, 2, M, 2, 2, M))
        A_tmp = math.einsum("...ijklmn->...jikmln", A_tmp)
        Aphi = math.reshape(A_tmp, (*batch_shape, 4 * M, 4 * M))

        bphi = math.zeros((*batch_shape, 4 * M), dtype=math.complex128)
        phi = Channel.from_bargmann(
            core_modes,
            core_modes,
            (Aphi, bphi, math.ones(batch_shape, dtype=math.complex128)),
        )
        renorm = phi.contract(TraceOut(self.modes))
        phi = phi / renorm.ansatz.c

        a = reduced_A[..., M:, M:]
        Acore = math.block(
            [
                [math.zeros((*batch_shape, M, M), dtype=math.complex128), r_core_transpose],
                [r_core, a],
            ],
        )
        bcore_m = math.einsum("...ij,...j->...i", math.inv(Gamma_phi_transpose), bm)
        bcore_m_ket = bcore_m[..., M:]
        bcore_n = bn - math.einsum("...ij,...jk,...k->...i", temp, Aphi_in, bcore_m)
        bcore_n_ket = bcore_n[..., (m_modes - M) :]

        inverse_order = np.argsort(core_ket_indices + other_ket_indices)
        Acore = Acore[..., inverse_order, :][..., :, inverse_order]
        bcore = math.concat([bcore_m_ket, bcore_n_ket], -1)[..., inverse_order]
        c_core = math.ones_like(c)

        core = Ket.from_bargmann(self.modes, (Acore, bcore, c_core))
        for i in range(M):
            core = core.contract(
                Dgate(
                    core_modes[i],
                    -math.real(bcore_m_ket[..., i]),
                    -math.imag(bcore_m_ket[..., i]),
                ),
                mode="zip",
            )
            dgate_u = Dgate(
                core_modes[i],
                math.real(bcore_m_ket[..., i]),
                math.imag(bcore_m_ket[..., i]),
            )
            dgate_ch = dgate_u.contract(dgate_u.adjoint, mode="zip")
            phi = dgate_ch.contract(phi, mode="zip")
        c_core = math.ones_like(c)
        phi = Channel.from_bargmann(core_modes, core_modes, (phi.ansatz.A, phi.ansatz.b, c_core))
        renorm = phi.contract(TraceOut(self.modes))
        phi = phi / renorm.ansatz.c
        return (
            Ket.from_bargmann(core.modes, (core.ansatz.A, core.ansatz.b, c_core)).normalize(),
            phi,
        )

    def physical_stellar_decomposition_mixed(
        self,
        core_modes: Collection[int],
    ) -> tuple[DM, Channel]:
        r"""
        Applies the physical stellar decomposition based on the rank condition.

        Args:
            core_modes: the core modes defining the core variables.

        Returns:
            core: The core state (`DM`)
            phi: The channel acting on the core modes (`Channel`)

        Raises:
            ValueError: If the rank condition is not satisfied.

        .. code-block::

            >>> from mrmustard.lab import DM, Vacuum

            >>> rho = DM.random([0,1])
            >>> core, phi = rho.physical_stellar_decomposition_mixed([0])

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

        new_order = math.astensor(core_indices + other_indices)

        A, b, c = self.ansatz.reorder(new_order).triple
        batch_shape = self.ansatz.batch_shape

        M = len(core_modes)
        N = self.n_modes - M

        Am = A[..., : 2 * M, : 2 * M]
        An = A[..., 2 * M :, 2 * M :]
        R = A[..., 2 * M :, : 2 * M]
        bm = b[..., : 2 * M]
        bn = b[..., 2 * M :]

        sigma = R[..., M:, :M]
        r = R[..., M:, M:]
        alpha_m = Am[..., M:, :M]
        alpha_n = An[..., N:, :N]
        a_n = An[..., N:, N:]

        r_transpose = math.einsum("...ij->...ji", r)
        sigma_transpose = math.einsum("...ij->...ji", sigma)
        R_transpose = math.einsum("...ij->...ji", R)

        rank = np.linalg.matrix_rank(
            r @ math.conj(r_transpose) + sigma @ math.conj(sigma_transpose),
        )
        if math.any(rank > M):
            raise ValueError(
                "The physical mixed stellar decomposition is not possible for this DM, "
                f"as the rank {rank} of the off-diagonal block of the Bargmann matrix is larger than the number "
                f"of core modes {M}.",
            )

        I2M = math.broadcast_to(
            math.eye(2 * M, dtype=math.complex128),
            (*batch_shape, 2 * M, 2 * M),
        )
        reduced_A = R @ math.inv(I2M - math.Xmat(M) @ Am) @ math.conj(R_transpose)

        # computing a low-rank r_c:
        r_c_squared = reduced_A[..., N:, N:] + sigma @ math.inv(alpha_m) @ math.conj(
            sigma_transpose,
        )
        r_c_evals, r_c_evecs = math.eigh(r_c_squared)
        r_c = math.einsum(
            "...ij,...j->...ij",
            r_c_evecs[..., -M:],
            math.sqrt(r_c_evals[..., -M:], dtype=math.complex128),
        )
        Os_NM = math.zeros((*batch_shape, N, M), dtype=math.complex128)
        Os_MN = math.zeros((*batch_shape, M, N), dtype=math.complex128)
        R_c = math.block([[math.conj(r_c), Os_NM], [Os_MN, r_c]])
        R_c_transpose = math.einsum("...ij->...ji", R_c)

        Aphi_out = Am
        gamma = math.pinv(R_c) @ R
        gamma_transpose = math.einsum("...ij->...ji", gamma)
        Aphi_in = gamma @ math.inv(Aphi_out - math.Xmat(M)) @ gamma_transpose + math.Xmat(M)

        Aphi_oi = math.block([[Aphi_out, gamma_transpose], [gamma, Aphi_in]])
        A_tmp = math.reshape(Aphi_oi, (*batch_shape, 2, 2, M, 2, 2, M))
        A_tmp = math.einsum("...ijklmn->...jikmln", A_tmp)
        Aphi = math.reshape(A_tmp, (*batch_shape, 4 * M, 4 * M))
        bphi = math.zeros((*batch_shape, 4 * M), dtype=math.complex128)
        c_phi = math.ones_like(c)
        phi = Channel.from_bargmann(core_modes, core_modes, (Aphi, bphi, c_phi))
        renorm = phi.contract(TraceOut(self.modes))
        phi = phi / renorm.ansatz.c

        alpha_core_n = alpha_n - sigma @ math.inv(alpha_m) @ math.conj(sigma_transpose)
        a_core_n = a_n + reduced_A[..., N:, :N]
        A_core_n = math.block(
            [[math.conj(a_core_n), math.conj(alpha_core_n)], [alpha_core_n, a_core_n]],
        )

        A_core = math.block(
            [
                [math.zeros((*batch_shape, 2 * M, 2 * M), dtype=math.complex128), R_c_transpose],
                [R_c, A_core_n],
            ],
        )
        b_core_m = math.einsum("...ij,...j->...i", math.inv(gamma_transpose), bm)
        b_core_n = bn - math.einsum("...ij,...jk,...k->...i", R_c, Aphi_in, b_core_m)

        b_core = math.concat([b_core_m, b_core_n], -1)
        inverse_order = np.argsort(new_order)
        A_core = A_core[..., inverse_order, :][..., inverse_order]
        b_core = b_core[..., inverse_order]
        core = DM.from_bargmann(
            self.modes,
            (A_core, b_core, c_phi),
        )
        alpha = b_core[..., core_ket_indices]
        for i, m in enumerate(core_modes):
            d_g = Dgate(m, -math.real(alpha[..., i]), -math.imag(alpha[..., i]))
            d_g_inv = d_g.inverse()
            d_ch = d_g.contract(d_g.adjoint, mode="zip")
            d_ch_inverse = d_g_inv.contract(d_g_inv.adjoint, mode="zip")

            core = core.contract(d_ch, mode="zip")
            phi = (d_ch_inverse).contract(phi, mode="zip")

        core = DM.from_bargmann(self.modes, (core.ansatz.A, core.ansatz.b, c_phi)).normalize()
        phi = Channel.from_bargmann(core_modes, core_modes, (phi.ansatz.A, phi.ansatz.b, c_phi))
        renorm = phi.contract(TraceOut(self.modes))
        phi = phi / renorm.ansatz.c
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
        return DM(ansatz, wires, name=self.name)

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

        .. code-block::

            >>> from mrmustard.lab import CircuitComponent, DM, TraceOut

            >>> assert isinstance(DM.random([0]).dual >> DM.random([0]), CircuitComponent)
            >>> assert isinstance(DM.random([0,1]) >> TraceOut(0), DM)
        """

        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input and result.wires.bra.modes == result.wires.ket.modes:
            return DM._from_attributes(result.ansatz, result.wires)
        return result
