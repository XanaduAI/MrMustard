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

"""This module contains the defintion of the density matrix class ``DM``."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from IPython.display import display

from mrmustard import math, settings, widgets
from mrmustard.physics import stellar
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.fock_utils import fidelity as fock_dm_fidelity
from mrmustard.physics.gaussian import fidelity as gaussian_fidelity
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import ComplexTensor, Matrix, Scalar, Vector

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import TraceOut
from ..transformations import Channel, Dgate, Map
from ..utils import shape_check
from .base import State
from .builtins import wigner_to_bargmann_rho

__all__ = ["DM"]


class DM(State):
    r"""Base class for density matrices."""

    short_name = "DM"

    @property
    def is_positive(self) -> bool:
        r"""Whether this DM corresponds to a positive operator.

        >>> from mrmustard.lab import GaussianDM
        >>> assert GaussianDM.random([0]).is_positive

        Raises:
            NotImplementedError: If the state is mixed.
            NotImplementedError: If the state has derived variables.
            NotImplementedError: If the state has an ``ArrayAnsatz``.
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
        r"""Whether this DM is a physical density operator.

        >>> from mrmustard.lab import GaussianDM
        >>> assert GaussianDM.random([0]).is_physical
        """
        return self.is_positive and math.allclose(self.probability, 1, settings.ATOL)

    @property
    def probability(self) -> float:
        r"""Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states.
        """
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices
        rep = self.ansatz.trace(idx_ket, idx_bra)
        return math.real(rep.scalar)

    @property
    def purity(self) -> float:
        r"""Computes the purity (:math:`tr(rho^2)`) of this DM.

        >>> from mrmustard import math
        >>> from mrmustard.lab import DM, Vacuum
        >>> assert math.allclose(Vacuum([0]).dm().purity, 1.0)
        """
        return self.L2_norm / self.probability**2

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        name: str | None = None,
        lin_sup: bool = False,
    ) -> DM:
        return super().from_bargmann(modes=modes, triple=triple, name=name, lin_sup=lin_sup)

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batch_dims: int = 0,
    ) -> DM:
        return super().from_fock(modes=modes, array=array, name=name, batch_dims=batch_dims)

    @classmethod
    def from_ansatz(
        cls,
        modes: Collection[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> DM:
        if not isinstance(modes, set) and sorted(modes) != list(modes):
            raise ValueError(f"Modes must be sorted. got {modes}")
        modes = set(modes)
        if ansatz and ansatz.core_dims != 2 * len(modes):
            raise ValueError(
                f"Expected an ansatz with {2 * len(modes)} variables, found {ansatz.core_dims}.",
            )
        wires = Wires(modes_out_bra=set(modes), modes_out_ket=set(modes))
        if ansatz is not None:
            ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz)
            if representation == ReprEnum.FOCK:
                for w in wires.quantum:
                    w.repr = ReprEnum.FOCK
                    w.fock_shape = ansatz.core_shape[w.index]
        else:
            ansatz_factory = None
        return DM(ansatz_factory=ansatz_factory, wires=wires, name=name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Collection[int],
        triple: tuple[Matrix, Vector, Scalar],
        name: str | None = None,
        atol_purity: float | None = None,
    ) -> DM:
        r"""Initializes a density matrix from the covariance matrix, vector of means and a coefficient,
        which parametrize the s-parametrized phase space function
        :math:`coeff * exp(-1/2(x-means)^T cov^{-1} (x-means))`.h:`coeff * exp((x-means)^T cov^{-1} (x-means))`.

        >>> from mrmustard import math
        >>> from mrmustard.lab import DM, Vacuum
        >>> rho = DM.from_phase_space([0], (math.eye(2)/2, [0,0], 1))
        >>> assert rho == Vacuum([0]).dm()

        Args:
            modes: The modes of this states.
            triple: The ``(cov, means, coeff)`` triple.
            name: The name of this state.
            atol_purity: Unused argument.

        Returns:
            A ``DM`` object from its phase space representation.

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

        ansatz_dict = {
            ReprEnum.BARGMANN: (wigner_to_bargmann_rho, ("cov", "means", "lin_sup")),
        }
        ansatz_factory = AnsatzFactory(ansatz_dict=ansatz_dict, cov=cov, means=means)
        ansatz = ansatz_factory(cov=cov, means=means, representation=ReprEnum.BARGMANN)
        return DM.from_ansatz(modes, coeff * ansatz, name)

    @classmethod
    def from_quadrature(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        phi: float = 0.0,
        name: str | None = None,
    ) -> DM:
        return super().from_quadrature(modes=modes, triple=triple, phi=phi, name=name)

    def dm(self) -> DM:
        r"""The ``DM`` object obtained from this ``DM``.

        >>> from mrmustard.lab import Vacuum, DM
        >>> assert isinstance(Vacuum([0]).dm(), DM)

        Returns:
            A ``DM``.
        """
        return self

    def expectation(self, operator: CircuitComponent) -> Scalar:
        r"""The expectation value of an operator with respect to this DM.

        Given the operator `O`, this function returns :math:`Tr\big(\rho O)`\, where :math:`\rho`
        is the density matrix of this state.

        The ``operator`` is expected to be a component with ket-like wires (i.e., output wires on
        the ket side), density matrix-like wires (output wires on both ket and bra sides), or
        unitary-like wires (input and output wires on the ket side).

        >>> from mrmustard import math
        >>> from mrmustard.lab import Rgate, GaussianDM
        >>> beta = 1
        >>> symplectic = math.eye(2)
        >>> rho = GaussianDM([0], beta, symplectic)
        >>> answer = (1-math.exp(-beta))/(1+math.exp(-beta))
        >>> assert math.allclose(rho.expectation(Rgate(0, np.pi)), answer)

        Args:
            operator: A ket-like, density-matrix like, or unitary-like circuit component.

        Returns:
            Expectation value either as a complex number or a batch of complex numbers.

        Raise:
            ValueError: If the operator is not a ket-like, density-matrix like, or unitary-like component.
            ValueError: If the operator is defined over a set of modes that is not a subset of the modes of this state.
            ValueError: If the modes in common between the operator and the state are not a subset of the modes of the operator.
        """
        if self.wires.modes.isdisjoint(operator.wires.modes):
            raise ValueError(f"No modes in common between {self} and {operator}.")

        self_rep = self >> TraceOut(self.wires.modes - operator.wires.modes)

        if operator.wires.is_ket_like:
            ret = operator.expectation(self_rep)

        elif operator.wires.is_dm_like:
            ret = self_rep.contract(operator.dual)

        elif operator.wires.is_unitary_like:
            if not operator.wires.modes.issubset(self.wires.modes):
                raise ValueError(
                    f"Unitary-like operator modes {operator.wires.modes} are not a subset of the state modes {self.wires.modes}"
                )
            ret = self_rep.contract(operator) >> TraceOut(operator.wires.modes)

        else:
            raise ValueError(f"Cannot calculate the expectation value of {operator} with {self}.")

        if isinstance(ret, CircuitComponent) and len(ret.wires.modes) == 0:
            return ret.ansatz.scalar

        return ret

    def fidelity(self, other: State) -> Scalar:
        r"""The fidelity between this DM and another ket or DM. If the other state is a Ket, fidelity
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
        r"""Returns an array representation of this component in the Fock basis with the given shape.

        The ``standard_order`` boolean argument lets one choose the standard convention for the
        index ordering of the density matrix. For a single mode, if ``standard_order=True`` the
        returned 2D array :math:`rho_{ij}` has a first index corresponding to the "left" (ket)
        side of the matrix and the second index to the "right" (bra) side. Otherwise, MrMustard's
        convention is that the bra index comes before the ket index. In other words, for a single
        mode, the array returned by ``fock_array`` with ``standard_order=False`` (false by default)
        is the transpose of the standard density matrix. For multiple modes, the same applies to
        each pair of indices of each mode.

        >>> from mrmustard import math
        >>> from mrmustard.lab import Vacuum, DM
        >>> assert math.allclose(Vacuum([0]).dm().fock_array(), math.astensor([[1]]))

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is generated via ``auto_shape``.

            standard_order: The ordering of the wires. If ``standard_order = False``, then the conventional ordering
                of bra-ket is chosen. However, if one wants to get the actual matrix representation in the
                standard conventions of linear algebra, then ``standard_order=True`` must be chosen.

        Returns:
            array: The Fock representation of this component.

        Raises:
            ValueError: If the shape is not valid for the component.

        Note:
            The ``standard_order`` boolean argument lets one choose the standard convention for the
            index ordering of the density matrix. For a single mode, if ``standard_order=True`` the
            returned 2D array :math:`rho_{ij}` has a first index corresponding to the "left" (ket)
            side of the matrix and the second index to the "right" (bra) side. Otherwise, MrMustard's
            convention is that the bra index comes before the ket index. In other words, for a single
            mode, the array returned by ``fock_array`` with ``standard_order=False`` (false by default)
            is the transpose of the standard density matrix. For multiple modes, the same applies to each
            pair of indices of each mode.
        """
        array = super().fock_array(shape)
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

    def formal_stellar_decomposition(self, core_modes: Collection[int]) -> tuple[DM, Map]:
        r"""Computes the formal stellar decomposition for the DM.

        >>> from mrmustard.lab import GaussianDM, Vacuum
        >>> rho = GaussianDM.random([0,1])
        >>> core, phi = rho.formal_stellar_decomposition([0])
        >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0).dm()
        >>> assert rho == core >> phi
        >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0).dm()

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            The core state (`DM`) and the Gaussian `Map` performing the stellar decomposition (not necessarily CPTP).

        Note:
            This method pulls out the map ``phi`` from the given state on the given modes, so that
            the remaining state is a core state. Formally, we have
            .. math::

                \rho = (\phi\otimes\mathcal I) \rho_{\mathrm{core}}

            where the map :math:`phi` acts on the given `core_modes` only.
            Core states have favorable properties in the Fock representation
            e.g., being sparse.
        """
        other_modes = [m for m in self.modes if m not in core_modes]
        idx = self.wires[core_modes].indices + self.wires[other_modes].indices
        inv = np.argsort(idx)
        M = len(core_modes)

        A, b, c = self.ansatz.A, self.ansatz.b, self.ansatz.c
        A = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2)
        b = math.gather(b, idx, axis=-1)

        core_tr, phi_tr = stellar.formal_stellar_triples((A, b, c), 2 * M)

        # reorder phi from out-in to standard order
        inv_phi = (
            list(range(M))
            + list(range(2 * M, 3 * M))
            + list(range(M, 2 * M))
            + list(range(3 * M, 4 * M))
        )

        A_core = math.gather(math.gather(core_tr[0], inv, axis=-1), inv, axis=-2)
        b_core = math.gather(core_tr[1], inv, axis=-1)

        A_phi = math.gather(math.gather(phi_tr[0], inv_phi, axis=-1), inv_phi, axis=-2)
        b_phi = math.gather(phi_tr[1], inv_phi, axis=-1)

        core = DM.from_bargmann(self.modes, (A_core, b_core, core_tr[2]))
        phi = Map.from_bargmann(core_modes, core_modes, (A_phi, b_phi, phi_tr[2]))

        return core, phi

    def _ket_stellar_decomposition(self, core_modes: Collection[int]):
        r"""Physical stellar decomposition returning a Ket core.

        This private method implements the algorithm that extracts a pure state (Ket) core
        from a DM. It only works when the number of core modes equals half the total modes.

        Args:
            core_modes: The core modes (must be exactly n_modes // 2).

        Returns:
            The core state (``Ket``) and the channel acting on the core modes (``Channel``).
        """
        from .ket import Ket  # noqa: PLC0415

        other_modes = [m for m in self.modes if m not in core_modes]
        core_modes = list(core_modes)
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
        M = len(core_modes)

        Am = A[..., : 2 * M, : 2 * M]
        An = A[..., 2 * M :, 2 * M :]
        bm = b[..., : 2 * M]
        bn = b[..., 2 * M :]
        R = A[..., 2 * M :, : 2 * M]
        R_transpose = math.einsum("...ij->...ji", R)

        # Computing the core state
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
            core = core.contract(Dgate(core_modes[i], -bcore_m_ket[..., i]))
            dgate_u = Dgate(core_modes[i], bcore_m_ket[..., i])
            dgate_ch = dgate_u.contract(dgate_u.adjoint)
            phi = dgate_ch.contract(phi)

        c_core = math.ones_like(c)
        phi = Channel.from_bargmann(core_modes, core_modes, (phi.ansatz.A, phi.ansatz.b, c_core))
        renorm = phi.contract(TraceOut(self.modes))
        phi = phi / renorm.ansatz.c

        return (
            Ket.from_bargmann(core.modes, (core.ansatz.A, core.ansatz.b, c_core)).normalize(),
            phi,
        )

    def physical_stellar_decomposition(self, core_modes: Collection[int]) -> tuple[State, Channel]:
        r"""Applies the physical stellar decomposition.

        When the number of core modes equals exactly half of the total modes (and n_modes is even),
        returns a ``Ket`` core and ``Channel``. Otherwise, returns a ``DM`` core and ``Channel``.

        >>> from mrmustard.lab import GaussianDM, Vacuum
        >>> rho = GaussianDM.random([0, 1])
        >>> core, phi = rho.physical_stellar_decomposition([0])
        >>> assert rho == core >> phi
        >>> assert (core >> Vacuum(1).dual).normalize() == Vacuum(0)

        Args:
            core_modes: The core modes defining the core variables.

        Returns:
            The core state (`DM` or `Ket` depending on the number of core modes) and the channel acting on the core modes.

        Raises:
            ValueError: If the rank condition is not satisfied (when core modes < half).
        """
        M = len(core_modes)
        n = self.n_modes

        # When M == n/2 and n is even, use the Ket decomposition
        if n // 2 == M and n % 2 == 0:
            return self._ket_stellar_decomposition(core_modes)

        # Otherwise, use the DM decomposition
        other_modes = [m for m in self.modes if m not in core_modes]
        idx = self.wires[core_modes].indices + self.wires[other_modes].indices
        inv = np.argsort(idx)

        A, b, c = self.ansatz.A, self.ansatz.b, self.ansatz.c
        A = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2)
        b = math.gather(b, idx, axis=-1)

        core_tr, phi_tr = stellar.physical_stellar_triples_dm((A, b, c), len(core_modes))

        A_core = math.gather(math.gather(core_tr[0], inv, axis=-1), inv, axis=-2)
        b_core = math.gather(core_tr[1], inv, axis=-1)

        core = DM.from_bargmann(self.modes, (A_core, b_core, core_tr[2]))
        phi = Channel.from_bargmann(core_modes, core_modes, phi_tr)

        return core, phi

    def _ipython_display_(self):  # pragma: no cover
        if widgets.IN_INTERACTIVE_SHELL:
            print(self)
            return
        is_fock = isinstance(self.ansatz, ArrayAnsatz)
        display(widgets.state(self, is_ket=False, is_fock=is_fock))

    def __rshift__(self, other: Scalar | CircuitComponent) -> Scalar | CircuitComponent:
        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input and result.wires.bra.modes == result.wires.ket.modes:
            return DM._from_attributes(result.ansatz, result.wires)
        return result
