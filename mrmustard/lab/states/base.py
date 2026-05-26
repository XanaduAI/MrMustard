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

"""The base for the ``State`` class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from itertools import product
from typing import Self

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from mrmustard import math, settings
from mrmustard.mathlib.lattice.autoshape import autoshape_numba
from mrmustard.mathlib.lattice.strategies.wormhole import (
    wormhole_1leftover_dm,
    wormhole_1leftover_ket,
)
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import bargmann_Abc_to_phasespace_cov_means
from mrmustard.physics.fock_utils import quadrature_distribution
from mrmustard.physics.gaussian import von_neumann_entropy
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexTensor, Matrix, RealVector, Scalar, Vector

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import BtoChar, BtoPS, BtoQ
from ..transformations import Transformation

__all__ = ["State"]


# ~~~~~~~
# Classes
# ~~~~~~~


class State(CircuitComponent):
    r"""Base class for all states."""

    @property
    def is_pure(self):
        r"""Whether this state is pure."""
        return math.allclose(self.purity, 1.0)

    @property
    def is_separable(self):
        r"""Check if a multi-mode quantum state is separable based on the von Neumann entropy.

        Returns:
            Whether the state is separable.

        Raises:
            NotImplementedError: If the state is a linear superposition.
        """
        if self.ansatz._lin_sup:
            raise NotImplementedError("Separation of linear superpositions is not implemented.")
        if self.n_modes == 1:
            return True
        rho = self.dm()
        cov_full, _, _ = rho.phase_space(s=0)
        S_total = von_neumann_entropy(cov_full)
        entropy_diff = -S_total
        for mode in self.modes:
            rho_reduced = rho.get_modes(mode)
            cov_reduced, _, _ = rho_reduced.phase_space(s=0)
            entropy = von_neumann_entropy(cov_reduced)
            entropy_diff += entropy
        return math.allclose(entropy_diff, 0)

    @property
    def L2_norm(self) -> float:
        r"""The `L2` norm squared of a ``Ket``, or the Hilbert-Schmidt norm of a ``DM``.

        >>> from mrmustard import math
        >>> from mrmustard.lab import GaussianKet
        >>> state = GaussianKet.random([0])
        >>> assert math.allclose(state.L2_norm, 1.0)
        """
        state = self
        if isinstance(state.ansatz, PolyExpAnsatz) and state.ansatz.num_derived_vars > 0:
            state = state.to_fock()
        return math.real(state.contract(state.dual).ansatz.scalar)

    @property
    @abstractmethod
    def probability(self) -> float:
        r"""Returns :math:`\langle\psi|\psi\rangle` for ``Ket`` states
        :math:`|\psi\rangle` and :math:`\text{Tr}(\rho)` for ``DM`` states :math:`\rho`.
        """

    @property
    @abstractmethod
    def purity(self) -> float:
        r"""The purity of this state."""

    @property
    def wigner(self):
        r"""Returns the Wigner function of this state in phase space as an ``Ansatz``.

        >>> import numpy as np
        >>> from mrmustard.lab import GaussianKet
        >>> state = GaussianKet.random([0])
        >>> x = np.linspace(-5, 5, 100)
        >>> assert np.all(state.wigner(x,0).real >= 0)
        """
        if isinstance(self.ansatz, PolyExpAnsatz):
            return (self >> BtoPS(self.modes, s=0)).ansatz.PS
        raise ValueError(
            "Wigner ansatz not implemented for Fock states. Consider calling ``.to_bargmann()`` first.",
        )

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        name: str | None = None,
        lin_sup: bool = False,
    ) -> Self:
        r"""Initializes a state of type ``cls`` from an ``(A, b, c)`` triple
        parametrizing the Ansatz in Bargmann representation.

        >>> from mrmustard.physics.ansatz import PolyExpAnsatz
        >>> from mrmustard.physics.triples import coherent_state_Abc
        >>> from mrmustard.lab.states.ket import Ket
        >>> modes = (0,)
        >>> triple = coherent_state_Abc(alpha=0.1)
        >>> coh = Ket.from_bargmann(modes, triple)
        >>> assert coh.modes == modes
        >>> assert coh.ansatz == PolyExpAnsatz(*triple)
        >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this state.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.
            lin_sup: Whether to include linear superposition axes in the batch dimensions.

        Returns:
            A ``State``.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.
        """
        return cls.from_ansatz(modes, PolyExpAnsatz(*triple, lin_sup=lin_sup), name)

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batch_dims: int = 0,
    ) -> Self:
        r"""Initializes a state of type ``cls`` from an array parametrizing the
        state in Fock representation.

        >>> from mrmustard.physics.ansatz import ArrayAnsatz
        >>> from mrmustard.lab import Coherent, Ket
        >>> array = Coherent(mode=0, alpha=0.1).to_fock().ansatz.array
        >>> coh = Ket.from_fock((0,), array, batch_dims=0)
        >>> assert coh.modes == (0,)
        >>> assert coh.ansatz == ArrayAnsatz(array, batch_dims=0)
        >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this state.
            array: The Fock array.
            name: The name of this state.
            batch_dims: The number of batch dimensions in the given array.

        Returns:
            A ``State``.

        Raises:
            ValueError: If the given array has a shape that is inconsistent with the number of
                modes.
        """
        return cls.from_ansatz(modes, ArrayAnsatz(array, batch_dims=batch_dims), name)

    @classmethod
    @abstractmethod
    def from_ansatz(
        cls,
        modes: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> Self:
        r"""Initializes a state of type ``cls`` given modes and an ansatz.

        >>> from mrmustard import math
        >>> from mrmustard.lab import Ket
        >>> from mrmustard.physics.ansatz import PolyExpAnsatz
        >>> A = math.astensor([[0,.5], [.5,0]])
        >>> b = math.astensor([2-1j,2+1j])
        >>> c = 1
        >>> psi = Ket.from_ansatz([0,1], PolyExpAnsatz(A,b,c))
        >>> assert isinstance(psi, Ket)

        Args:
            modes: The modes of this state.
            ansatz: The ansatz of this state.
            name: The name of this state.

        Returns:
            A state.
        """

    @classmethod
    @abstractmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        name: str | None = None,
        atol_purity: float | None = None,
    ) -> Self:
        r"""Initializes a state from the covariance matrix and the vector of means of a state in
        phase space.

        >>> from mrmustard import math
        >>> from mrmustard.lab import Ket, Vacuum
        >>> assert Ket.from_phase_space([0], (math.eye(2)/2, [0,0], 1)) == Vacuum([0])

        Note:
            If the given covariance matrix and vector of means are consistent with a pure
            state, a ``Ket`` is returned. Otherwise, a ``DM`` is returned. One can skip this check by
            setting ``atol_purity`` to ``None`` (``atol_purity`` defaults to ``None``).

        Args:
            modes: The modes of this states.
            triple: A covariance matrix, vector of means, and constant multiple triple.
            name: The name of this state.
            atol_purity: If ``atol_purity`` is given, the purity of the state is computed, and an
                error is raised if its value is smaller than ``1-atol_purity`` or larger than
                ``1+atol_purity``. If ``None``, this check is skipped.

        Returns:
            A ``State``.

        Raises:
            ValueError: If the given ``cov`` and ``means`` have shapes that are inconsistent
                with the number of modes.
            ValueError: If ``atol_purity`` is not ``None`` and the purity of the returned state
                is smaller than ``1-atol_purity`` or larger than ``1+atol_purity``.
        """

    @classmethod
    def from_quadrature(
        cls,
        modes: Sequence[int],
        triple: tuple[Matrix, Vector, Scalar],
        phi: float = 0.0,
        name: str | None = None,
    ) -> Self:
        r"""Initializes a state from a triple (A,b,c) that parametrizes the wavefunction
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
        Q = cls.from_ansatz(modes, PolyExpAnsatz(*triple))
        return cls.from_ansatz(modes, (Q >> QtoB).ansatz, name)

    def auto_shape(
        self,
        max_prob=None,
        max_shape=None,
        min_shape=None,
        respect_manual_shape=True,
    ) -> tuple[int, ...]:
        r"""Generates an estimate for the Fock shape. If the state is in Fock the core shape is used.
        If in Bargmann, the shape is computed as the shape that captures at least ``settings.AUTOSHAPE_PROBABILITY``
        of the probability mass of each single-mode marginal (default 99.9%) so long as the state has no derived variables
        and is unbatched. Otherwise, defaults to ``settings.DEFAULT_FOCK_SIZE``. If ``respect_manual_shape`` is ``True``,
        the non-None values in ``self.manual_shape`` are used to override the shape.

        >>> from mrmustard import math
        >>> from mrmustard.lab import Vacuum
        >>> assert math.allclose(Vacuum([0]).fock_array(), 1)

        Note:
            If jitted, the shape will default to ``settings.DEFAULT_FOCK_SIZE``.

        Args:
            max_prob: The maximum probability mass to capture in the shape. Default is ``settings.AUTOSHAPE_PROBABILITY``.
            max_shape: The maximum shape cutoff. Default is ``settings.AUTOSHAPE_MAX``.
            min_shape: The minimum shape cutoff. Default is ``settings.AUTOSHAPE_MIN``.
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``. Default is ``True``.

        Returns:
            The Fock shape of this component.
        """
        try:
            shape = self.ansatz.core_shape
        except AttributeError:
            if self.ansatz.num_derived_vars == 0 and self.ansatz.batch_dims == 0:
                if not self.wires.ket or not self.wires.bra:
                    ansatz = self.ansatz.conj & self.ansatz
                else:
                    ansatz = self.ansatz
                A, b, c = ansatz.triple
                try:
                    shape = autoshape_numba(
                        math.asnumpy(A),
                        math.asnumpy(b),
                        math.asnumpy(c),
                        max_prob or settings.AUTOSHAPE_PROBABILITY,
                        max_shape or settings.AUTOSHAPE_MAX,
                        min_shape or settings.AUTOSHAPE_MIN,
                    )
                # covers the case where auto_shape is jitted
                except math.BackendError:  # pragma: no cover
                    shape = super().auto_shape()
                if self.wires.ket and self.wires.bra:
                    shape = tuple(shape) + tuple(shape)
            else:
                shape = super().auto_shape()
        if respect_manual_shape:
            return tuple(c or s for c, s in zip(self.manual_shape, shape))
        return tuple(shape)

    def fock_distribution(self, cutoff: int) -> ComplexTensor:
        r"""Returns the Fock distribution of the state up to some cutoff.

        Args:
            cutoff: The photon cutoff (maximum photon number).

        Returns:
            The Fock distribution including states from :math:`|0\rangle` to :math:`|\text{cutoff}\rangle`.
        """
        batch_shape = self.ansatz.batch_shape
        batch_dim = self.ansatz.batch_dims
        fock_array = self.fock_array(cutoff + 1)
        if not self.wires.ket or not self.wires.bra:
            return math.reshape(math.abs(fock_array) ** 2, (*batch_shape, -1))
        n_modes = self.n_modes
        if self.is_separable:
            for i in range(n_modes):
                fock_array = math.diagonal(fock_array, axis1=-n_modes - 1, axis2=-(1 + i))
            return math.reshape(math.abs(fock_array), (*batch_shape, -1))
        indices_list = [(...,) + ns * 2 for ns in product(list(range(cutoff + 1)), repeat=n_modes)]
        return math.stack([fock_array[indices] for indices in indices_list], axis=batch_dim)

    def get_modes(self, modes: int | Sequence[int]) -> State:
        r"""Reduced density matrix obtained by tracing out all the modes except those in
        ``modes``. Note that the result is returned with modes in increasing order.

        Args:
            modes: The modes to keep.

        Returns:
            A ``State`` object with the remaining modes.

        Raises:
            ValueError: If the modes to keep are not a subset of the modes of the state.
        """
        if not self.wires.ket or not self.wires.bra:
            return self.dm().get_modes(modes)
        keep = {modes} if isinstance(modes, int) else set(modes)
        if not keep.issubset(self.modes):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{keep}``.")
        idxz = [i for i, m in enumerate(self.modes) if m not in keep]
        idxz_conj = [i + len(self.modes) for i, m in enumerate(self.modes) if m not in keep]
        ansatz = self.ansatz.trace(idxz, idxz_conj)
        return self._from_attributes(
            ansatz, Wires(modes_out_bra=keep, modes_out_ket=keep), self.name
        )

    def wormhole_1mode(
        self,
        pnr_outcomes: tuple[int, ...] | list[tuple[int, ...]],
        output_cutoff: int,
        leftover_mode: int,
    ) -> dict[tuple[int, ...], State]:
        r"""Compute conditional single-mode states given PNR measurements.

        Uses the wormhole algorithm to efficiently compute the conditional state
        of one "leftover" mode given photon number resolving (PNR) measurements
        on all other modes. This is much more efficient than computing the full
        Fock tensor and slicing for high photon counts.

        Supports both single and batched states. For batched states, the computation
        is parallelized across batch elements for improved performance.

        Args:
            pnr_outcomes: Either a single tuple or a list of tuples specifying photon
                counts for the measured modes. Each tuple has length n_modes - 1,
                with photon counts in mode order (excluding the leftover mode).
            output_cutoff: Maximum photon number for the output state.
            leftover_mode: The mode to keep unmeasured. Must be one of this state's modes.

        Returns:
            Dict mapping PNR tuples to State objects (same type as self).
            For batched input states, the returned states will also be batched.

        Raises:
            ValueError: If this state has fewer than 2 modes.
            ValueError: If leftover_mode is not one of this state's modes.
            ValueError: If pnr_outcomes tuples have wrong length.

        Example:
            >>> import numpy as np
            >>> from mrmustard.lab import GaussianKet
            >>> ket = GaussianKet.random([0, 1], seed=42)
            >>> pnr = (2,)
            >>> cutoff = 5
            >>> results = ket.wormhole_1mode(pnr, output_cutoff=cutoff, leftover_mode=0)
            >>> cond_ket = results[pnr]
            >>> cond_ket.fock_array().shape
            (6,)

        Note:
            The wormhole algorithm is particularly efficient for high photon counts
            where computing the full Fock tensor would be prohibitively expensive.
            Memory scaling depends on state type: O(2^M × cutoff) for Ket,
            O(4^M × cutoff²) for DM, where M is the number of measured modes.
        """
        if self.n_modes < 2:
            raise ValueError("wormhole_1mode requires at least 2 modes")

        if isinstance(pnr_outcomes, tuple):
            pnr_outcomes = [pnr_outcomes]

        modes = sorted(self.modes)

        if leftover_mode not in self.modes:
            raise ValueError(f"leftover_mode {leftover_mode} is not in this state's modes {modes}")

        expected_len = self.n_modes - 1
        for pnr in pnr_outcomes:
            if len(pnr) != expected_len:
                raise ValueError(
                    f"Each PNR tuple must have length {expected_len} (n_modes - 1), got {len(pnr)}"
                )

        A, b, c = self.bargmann_triple()
        leftover_mode_idx = modes.index(leftover_mode)

        if not self.wires.ket or not self.wires.bra:
            # Ket case
            results_arrays = wormhole_1leftover_ket(
                A,
                b,
                c,
                output_cutoff=output_cutoff,
                pnr_outcomes=pnr_outcomes,
                leftover_mode=leftover_mode_idx,
            )
        else:
            # DM case
            results_arrays = wormhole_1leftover_dm(
                A,
                b,
                c,
                output_cutoff=output_cutoff,
                pnr_outcomes=pnr_outcomes,
                leftover_mode=leftover_mode_idx,
            )

        # Determine batch dimensions from input state
        batch_dims = self.ansatz.batch_dims

        results: dict[tuple[int, ...], State] = {}
        for pnr_tuple, state_array in results_arrays.items():
            state_obj = self.__class__.from_ansatz(
                modes=[leftover_mode],
                ansatz=ArrayAnsatz(state_array, batch_dims=batch_dims),
            )
            results[pnr_tuple] = state_obj

        return results

    @abstractmethod
    def formal_stellar_decomposition(
        self,
        core_modes: Sequence[int],
    ) -> tuple[State, Transformation]:
        r"""Applies the formal stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            A tuple containing the core state and the Gaussian transformation performing the stellar decomposition.
        """

    def normalize(self) -> State:
        r"""Returns a rescaled version of the state such that its probability is 1."""
        probability = self.probability
        if not self.wires.ket or not self.wires.bra:
            return self / math.sqrt(probability)
        return self / probability

    def phase_space(self, s: float) -> tuple:
        r"""Returns the phase space parametrization of a state, consisting in a covariance matrix, a vector of means
        and a scaling coefficient. When a state is a linear superposition of Gaussians, each of cov, means,
        coeff are arranged in a batch.

        Phase space representations are labelled by an ``s`` parameter (float) which modifies the
        exponent of :math:`D_s(\gamma) = e^{\frac{s}{2}|\gamma|^2}D(\gamma)`, which is the operator
        basis used to expand phase space density matrices.

        The ``s`` parameter typically takes the values of -1, 0, 1 to indicate Glauber/Wigner/Husimi functions.

        Args:
            s: The phase space parameter

        Returns:
            The covariance matrix, the mean vector and the coefficient of the state in s-parametrized phase space.
        """
        if not isinstance(self.ansatz, PolyExpAnsatz):
            raise ValueError("Can calculate phase space only for Bargmann states.")

        if not self.wires.ket or not self.wires.bra:
            new_state = self.adjoint.contract(self.contract(BtoChar(self.modes, s=s)))
        else:
            new_state = self.contract(BtoChar(self.modes, s=s))
        return bargmann_Abc_to_phasespace_cov_means(*new_state.bargmann_triple())

    @abstractmethod
    def physical_stellar_decomposition(
        self,
        core_modes: Sequence[int],
    ) -> tuple[State, Transformation]:
        r"""Applies the physical stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            A tuple containing the core state and the Gaussian transformation performing the stellar decomposition.
        """

    def quadrature_distribution(self, *quad: RealVector, phi: float = 0.0) -> ComplexTensor:
        r"""The (discretized) quadrature distribution of the ``State``.

        Args:
            quad: the discretized quadrature axis over which the distribution is computed.
            phi: The quadrature angle. ``0`` corresponds to the x quadrature, ``pi/2`` to the p quadrature.

        Returns:
            The quadrature distribution.
        """
        if len(quad) != 1 and len(quad) != self.n_modes:
            raise ValueError(
                f"Expected {self.n_modes} or ``1`` quadrature vectors, got {len(quad)}.",
            )
        if len(quad) == 1:
            quad = quad * self.n_modes
        if not self.wires.ket or not self.wires.bra:
            return math.abs(self.quadrature(*quad, phi=phi)) ** 2
        return math.abs(self.quadrature(*(quad * 2), phi=phi))

    def visualize_2d(
        self,
        xbounds: tuple[int, int] = (-6, 6),
        pbounds: tuple[int, int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "RdBu",
        return_fig: bool = False,
        min_shape: int = 50,
    ) -> go.Figure | None:
        r"""2D visualization of the Wigner function of this state.

        Plots the Wigner function on a heatmap, alongside the probability distributions on the
        two quadrature axis.

        >>> from mrmustard.lab import Coherent
        >>> state = Coherent(0, alpha=1) / 2**0.5 + Coherent(0, alpha=-1) / 2**0.5
        >>> # state.visualize_2d()

        Args:
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``\'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.
            min_shape: The minimum fock shape to use for the Wigner function plot.

        Returns:
            A ``Plotly`` figure representing the state in 2D.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes > 1:
            raise ValueError("2D visualization not available for multi-mode states.")
        if self.ansatz.batch_dims > 1:
            raise NotImplementedError("2D visualization not implemented for batched states.")

        shape = [max(min_shape, d) for d in self.auto_shape()]
        dm = self.to_fock(tuple(shape)).dm().ansatz.array

        x, prob_x = quadrature_distribution(dm)
        p, prob_p = quadrature_distribution(dm, np.pi / 2)

        mask_x = math.asnumpy([xi >= xbounds[0] and xi <= xbounds[1] for xi in x])
        x = x[mask_x]
        prob_x = prob_x[mask_x]

        mask_p = math.asnumpy([pi >= pbounds[0] and pi <= pbounds[1] for pi in p])
        p = p[mask_p]
        prob_p = prob_p[mask_p]

        xvec = np.linspace(*xbounds, resolution)
        pvec = np.linspace(*pbounds, resolution)
        z, xs, ps = wigner_discretized(dm, xvec, pvec)
        xs = xs[:, 0]
        ps = ps[0, :]

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[5, 3],
            row_heights=[1, 2],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            shared_xaxes="columns",
            shared_yaxes="rows",
        )

        # X-P plot
        # note: heatmaps revert the y axes, which is why the minus in `y=-ps` is required
        fig_21 = go.Heatmap(
            x=xs,
            y=-ps,
            z=math.transpose(z),
            coloraxis="coloraxis",
            name="Wigner function",
            autocolorscale=False,
        )
        fig.add_trace(fig_21, row=2, col=1)
        fig.update_traces(row=2, col=1)
        fig.update_xaxes(range=xbounds, title_text="x", row=2, col=1)
        fig.update_yaxes(range=pbounds, title_text="p", row=2, col=1)

        # X quadrature probability distribution
        fig_11 = go.Scatter(x=x, y=prob_x, line={"color": "steelblue", "width": 2}, name="Prob(x)")
        fig.add_trace(fig_11, row=1, col=1)
        fig.update_xaxes(range=xbounds, row=1, col=1, showticklabels=False)
        fig.update_yaxes(title_text="Prob(x)", range=(0, max(prob_x)), row=1, col=1)

        # P quadrature probability distribution
        fig_22 = go.Scatter(x=prob_p, y=p, line={"color": "steelblue", "width": 2}, name="Prob(p)")
        fig.add_trace(fig_22, row=2, col=2)
        fig.update_xaxes(title_text="Prob(p)", range=(0, max(prob_p)), row=2, col=2)
        fig.update_yaxes(range=pbounds, row=2, col=2, showticklabels=False)

        fig.update_layout(
            height=500,
            width=580,
            plot_bgcolor="aliceblue",
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            showlegend=False,
            coloraxis={"colorscale": colorscale, "cmid": 0},
        )
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            tickfont_family="Arial Black",
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            tickfont_family="Arial Black",
        )

        if return_fig:
            return fig
        display(fig)
        return None

    def visualize_2d_with_arrows(
        self,
        arrows: np.ndarray,
        xbounds: tuple[int, int] = (-6, 6),
        pbounds: tuple[int, int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "RdBu",
        return_fig: bool = False,
        min_shape: int = 50,
    ) -> go.Figure | None:
        r"""Plot the state Wigner function and q/p marginals along
        with arrows from the origin of the Wigner function.

        Useful for, e.g., visualizing the stabilizer arguments of a GKP state.

        Args:
            arrows: 1D numpy array of complex numbers representing arrow end-points.
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.
            min_shape: The minimum fock shape to use for the Wigner function plot.

        Returns:
            A ``Plotly`` figure.
        """

        def _plot_arrow(x: float, y: float) -> go.Scatter:
            return go.Scatter(
                x=[0, x],
                y=[0, y],
                line={"color": "black", "width": 2},
                marker={"symbol": "arrow", "angleref": "previous", "size": 15},
            )

        fig = self.visualize_2d(
            xbounds=xbounds,
            pbounds=pbounds,
            resolution=resolution,
            colorscale=colorscale,
            return_fig=True,
            min_shape=min_shape,
        )
        assert fig is not None

        stabilizer_arrows = [
            _plot_arrow(
                alpha.real * np.sqrt(2 * settings.HBAR),
                alpha.imag * np.sqrt(2 * settings.HBAR),
            )
            for alpha in arrows
        ]

        for arrow in stabilizer_arrows:
            fig.add_trace(arrow, row=2, col=1)  # row=2 col=1 is the Wigner plot of the figure

        if return_fig:
            return fig
        display(fig)
        return None

    def visualize_3d(
        self,
        xbounds: tuple[int, int] = (-6, 6),
        pbounds: tuple[int, int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "RdBu",
        return_fig: bool = False,
        min_shape: int = 50,
    ) -> go.Figure | None:
        r"""3D visualization of the Wigner function of this state on a surface plot.

        Args:
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``\'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.
            min_shape: The minimum fock shape to use for the Wigner function plot.

        Returns:
            A ``Plotly`` figure representing the state in 3D.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes != 1:
            raise ValueError("3D visualization not available for multi-mode states.")
        if self.ansatz.batch_dims > 1:
            raise NotImplementedError("3D visualization not implemented for batched states.")
        shape = [max(min_shape, d) for d in self.auto_shape()]
        dm = self.to_fock(tuple(shape)).dm().ansatz.array
        xvec = np.linspace(*xbounds, resolution)
        pvec = np.linspace(*pbounds, resolution)
        z, xs, ps = wigner_discretized(dm, xvec, pvec)
        xs = xs[:, 0]
        ps = ps[0, :]

        fig = go.Figure(
            data=go.Surface(
                x=xs,
                y=ps,
                z=z,
                coloraxis="coloraxis",
                hovertemplate="x: %{x:.3f}<br>p: %{y:.3f}<br>W(x, p): %{z:.3f}<extra></extra>",
            ),
        )

        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            scene_camera_eye={"x": -2.1, "y": 0.88, "z": 0.64},
            coloraxis={"colorscale": colorscale, "cmid": 0},
        )
        fig.update_traces(
            contours_z={
                "show": True,
                "usecolormap": True,
                "highlightcolor": "limegreen",
                "project_z": False,
            },
        )
        fig.update_traces(
            contours_y={
                "show": True,
                "usecolormap": True,
                "highlightcolor": "red",
                "project_y": False,
            },
        )
        fig.update_traces(
            contours_x={
                "show": True,
                "usecolormap": True,
                "highlightcolor": "yellow",
                "project_x": False,
            },
        )
        fig.update_scenes(
            xaxis_title_text="x",
            yaxis_title_text="p",
            zaxis_title_text="Wigner function",
        )
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title="p")

        if return_fig:
            return fig
        display(fig)
        return None

    def visualize_dm(
        self,
        cutoff: int | None = None,
        return_fig: bool = False,
    ) -> go.Figure | None:
        r"""Plots the absolute value :math:`abs(\rho)` of the density matrix :math:`\rho` of this state
        on a heatmap.

        Args:
            cutoff: The desired cutoff. Defaults to the value of auto_shape.
            return_fig: Whether to return the ``Plotly`` figure.

        Returns:
            A ``Plotly`` figure representing absolute value of the density matrix of this state.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes != 1:
            raise ValueError("DM visualization not available for multi-mode states.")
        if self.ansatz.batch_dims > 1:
            raise NotImplementedError("DM visualization not implemented for batched states.")
        dm = self.to_fock(cutoff).dm().ansatz.array
        fig = go.Figure(
            data=go.Heatmap(z=abs(dm), colorscale="viridis", name="abs(ρ)", showscale=False),
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=257,
            width=257,
            margin={"l": 30, "r": 30, "t": 30, "b": 20},
        )
        fig.update_xaxes(title_text=f"abs(ρ), cutoff={dm.shape[0]}")

        if return_fig:
            return fig
        display(fig)
        return None
