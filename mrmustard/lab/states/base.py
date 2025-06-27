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

"The base for the ``State`` class"

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from mrmustard import math, settings
from mrmustard.math.lattice.autoshape import autoshape_numba
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import bargmann_Abc_to_phasespace_cov_means
from mrmustard.physics.fock_utils import quadrature_distribution
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector, RealVector

from ..circuit_components import CircuitComponent
from ..circuit_components_utils import BtoChar, BtoPS, BtoQ
from ..transformations import Transformation

__all__ = ["State"]

# ~~~~~~~
# Helpers
# ~~~~~~~


class OperatorType(Enum):
    r"""
    A convenience Enum class used to tag the type operators in the ``expectation`` method
    of ``Ket``\s and ``DM``\s.
    """

    KET_LIKE = 1
    DM_LIKE = 2
    UNITARY_LIKE = 3
    INVALID_TYPE = 4


def _validate_operator(operator: CircuitComponent) -> tuple[OperatorType, str]:
    r"""
    A function used to validate an operator inside the ``expectation`` method of ``Ket`` and
    ``DM``.

    If ``operator`` is ket-like, density matrix-like, or unitary-like, returns the corresponding
    ``OperatorType`` and an empty string. Otherwise, it returns ``INVALID_TYPE`` and an error
    message.
    """
    w = operator.wires

    # check if operator is ket-like
    if w.ket.output and not w.ket.input and not w.bra:
        return (
            OperatorType.KET_LIKE,
            "",
        )

    # check if operator is density matrix-like
    if w.ket.output and w.bra.output and not w.ket.input and not w.bra.input:
        if w.ket.output.modes != w.bra.output.modes:
            msg = "Found DM-like operator with different modes for ket and bra wires."
            return OperatorType.INVALID_TYPE, msg
        return OperatorType.DM_LIKE, ""

    # check if operator is unitary-like
    if w.ket.input and w.ket.output and not w.bra.input and not w.bra.input:
        if w.ket.input.modes != w.ket.output.modes:
            msg = "Found unitary-like operator with different modes for input and output wires."
            return OperatorType.INVALID_TYPE, msg
        return OperatorType.UNITARY_LIKE, ""

    msg = "Cannot calculate the expectation value of the given ``operator``."
    return OperatorType.INVALID_TYPE, msg


# ~~~~~~~
# Classes
# ~~~~~~~


class State(CircuitComponent):
    r"""
    Base class for all states.
    """

    @property
    def is_pure(self):
        r"""
        Whether this state is pure.
        """
        return math.allclose(self.purity, 1.0)

    @property
    def L2_norm(self) -> float:
        r"""
        The `L2` norm squared of a ``Ket``, or the Hilbert-Schmidt norm of a ``DM``.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket

            >>> state = Ket.random([0])
            >>> assert math.allclose(state.L2_norm, 1.0)
        """
        if isinstance(self.ansatz, PolyExpAnsatz) and self.ansatz.num_derived_vars > 0:
            fock_state = self.to_fock()
            ret = math.real(fock_state.contract(fock_state.dual, mode="zip").ansatz.scalar)
        else:
            ret = math.real(self.contract(self.dual, mode="zip").ansatz.scalar)
        return ret

    @property
    @abstractmethod
    def probability(self) -> float:
        r"""
        Returns :math:`\langle\psi|\psi\rangle` for ``Ket`` states
        :math:`|\psi\rangle` and :math:`\text{Tr}(\rho)` for ``DM`` states :math:`\rho`.
        """

    @property
    @abstractmethod
    def purity(self) -> float:
        r"""
        The purity of this state.
        """

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
        lin_sup: bool = False,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` from an ``(A, b, c)`` triple
        parametrizing the Ansatz in Bargmann representation.

        Args:
            modes: The modes of this state.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A ``State``.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.

        .. code-block::

            >>> from mrmustard.physics.ansatz import PolyExpAnsatz
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab.states.ket import Ket

            >>> modes = (0,)
            >>> triple = coherent_state_Abc(x=0.1)

            >>> coh = Ket.from_bargmann(modes, triple)
            >>> assert coh.modes == modes
            >>> assert coh.ansatz == PolyExpAnsatz(*triple)
            >>> assert isinstance(coh, Ket)
        """
        return cls.from_ansatz(modes, PolyExpAnsatz(*triple, lin_sup=lin_sup), name)

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batch_dims: int = 0,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` from an array parametrizing the
        state in Fock representation.


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

        .. code-block::

            >>> from mrmustard.physics.ansatz import ArrayAnsatz
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab import Coherent, Ket

            >>> array = Coherent(mode=0, x=0.1).to_fock().ansatz.array
            >>> coh = Ket.from_fock((0,), array, batch_dims=0)

            >>> assert coh.modes == (0,)
            >>> assert coh.ansatz == ArrayAnsatz(array, batch_dims=0)
            >>> assert isinstance(coh, Ket)
        """
        return cls.from_ansatz(modes, ArrayAnsatz(array, batch_dims=batch_dims), name)

    @classmethod
    @abstractmethod
    def from_ansatz(
        cls,
        modes: Sequence[int],
        ansatz: PolyExpAnsatz | ArrayAnsatz | None = None,
        name: str | None = None,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` given modes and an ansatz.

        Args:
            modes: The modes of this state.
            ansatz: The ansatz of this state.
            name: The name of this state.

        Returns:
            A state.


        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket
            >>> from mrmustard.physics.ansatz import PolyExpAnsatz

            >>> A = math.astensor([[0,.5], [.5,0]])
            >>> b = math.astensor([2-1j,2+1j])
            >>> c = 1
            >>> psi = Ket.from_ansatz([0,1], PolyExpAnsatz(A,b,c))

            >>> assert isinstance(psi, Ket)
        """

    @classmethod
    @abstractmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
        atol_purity: float | None = None,
    ) -> State:
        r"""
        Initializes a state from the covariance matrix and the vector of means of a state in
        phase space.


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

        Note:
            If the given covariance matrix and vector of means are consistent with a pure
            state, a ``Ket`` is returned. Otherwise, a ``DM`` is returned. One can skip this check by
            setting ``atol_purity`` to ``None`` (``atol_purity`` defaults to ``None``).

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Ket, Vacuum

            >>> assert Ket.from_phase_space([0], (math.eye(2)/2, [0,0], 1)) == Vacuum([0])
        """

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
        Q = cls.from_ansatz(modes, PolyExpAnsatz(*triple))
        return cls.from_ansatz(modes, (Q >> QtoB).ansatz, name)

    def auto_shape(
        self,
        max_prob=None,
        max_shape=None,
        min_shape=None,
        respect_manual_shape=True,
    ) -> tuple[int, ...]:
        r"""
        Generates an estimate for the Fock shape. If the state is in Fock the core shape is used.
        If in Bargmann, the shape is computed as the shape that captures at least ``settings.AUTOSHAPE_PROBABILITY``
        of the probability mass of each single-mode marginal (default 99.9%) so long as the state has no derived variables
        and is unbatched. Otherwise, defaults to ``settings.DEFAULT_FOCK_SIZE``. If ``respect_manual_shape`` is ``True``,
        the non-None values in ``self.manual_shape`` are used to override the shape.

        Args:
            max_prob: The maximum probability mass to capture in the shape. Default is ``settings.AUTOSHAPE_PROBABILITY``.
            max_shape: The maximum shape cutoff. Default is ``settings.AUTOSHAPE_MAX``.
            min_shape: The minimum shape cutoff. Default is ``settings.AUTOSHAPE_MIN``.
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``. Default is ``True``.

        Returns:
            The Fock shape of this component.


        Note:
            If jitted, the shape will default to ``settings.DEFAULT_FOCK_SIZE``.
        Example:
        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.lab import Vacuum

            >>> assert math.allclose(Vacuum([0]).fock_array(), 1)
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
        r"""
        Returns the Fock distribution of the state up to some cutoff.

        Args:
            cutoff: The photon cutoff.

        Returns:
            The Fock distribution.
        """
        fock_array = self.fock_array(cutoff)
        if not self.wires.ket or not self.wires.bra:
            return math.reshape(math.abs(fock_array) ** 2, (-1,))
        return math.reshape(math.abs(math.diag_part(fock_array)), (-1,))

    @abstractmethod
    def formal_stellar_decomposition(
        self,
        core_modes: Sequence[int],
    ) -> tuple[State, Transformation]:
        r"""
        Applies the formal stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            state: The core state.
            transformation: The Gaussian transformation performing the stellar decomposition.
        """

    def normalize(self) -> State:
        r"""
        Returns a rescaled version of the state such that its probability is 1.
        """
        probability = self.probability
        if probability.shape != () and isinstance(self.ansatz, PolyExpAnsatz):
            delta = len(self.ansatz.c.shape) - len(probability.shape)
            probability = math.reshape(probability, probability.shape + (1,) * delta)
        elif probability.shape != () and isinstance(self.ansatz, ArrayAnsatz):
            probability = math.reshape(
                probability,
                probability.shape + (1,) * self.ansatz.core_dims,
            )
        if not self.wires.ket or not self.wires.bra:
            return self / math.sqrt(probability)
        return self / probability

    def phase_space(self, s: float) -> tuple:
        r"""
        Returns the phase space parametrization of a state, consisting in a covariance matrix, a vector of means and a scaling coefficient. When a state is a linear superposition of Gaussians, each of cov, means, coeff are arranged in a batch.
        Phase space representations are labelled by an ``s`` parameter (float) which modifies the exponent of :math:`D_s(\gamma) = e^{\frac{s}{2}|\gamma|^2}D(\gamma)`, which is the operator basis used to expand phase space density matrices.
        The ``s`` parameter typically takes the values of -1, 0, 1 to indicate Glauber/Wigner/Husimi functions.

        Args:
            s: The phase space parameter

            Returns:
                The covariance matrix, the mean vector and the coefficient of the state in s-parametrized phase space.
        """
        if not isinstance(self.ansatz, PolyExpAnsatz):
            raise ValueError("Can calculate phase space only for Bargmann states.")

        if not self.wires.ket or not self.wires.bra:
            new_state = self.adjoint.contract(self.contract(BtoChar(self.modes, s=s), "zip"), "zip")
        else:
            new_state = self.contract(BtoChar(self.modes, s=s), "zip")
        return bargmann_Abc_to_phasespace_cov_means(*new_state.bargmann_triple())

    @abstractmethod
    def physical_stellar_decomposition(
        self,
        core_modes: Sequence[int],
    ) -> tuple[State, Transformation]:
        r"""
        Applies the physical stellar decomposition.

        Args:
            core_modes: The set of modes defining core variables.

        Returns:
            state: The core state.
            transformation: The Gaussian transformation performing the stellar decomposition.
        """

    def quadrature_distribution(self, *quad: RealVector, phi: float = 0.0) -> ComplexTensor:
        r"""
        The (discretized) quadrature distribution of the State.

        Args:
            quad: the discretized quadrature axis over which the distribution is computed.
            phi: The quadrature angle. ``phi=0`` corresponds to the x quadrature,
                    ``phi=pi/2`` to the p quadrature. The default value is ``0``.
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
        r"""
        2D visualization of the Wigner function of this state.

        Plots the Wigner function on a heatmap, alongside the probability distributions on the
        two quadrature axis.

        .. code-block::

            >>> from mrmustard.lab import Coherent

            >>> state = Coherent(0, x=1) / 2**0.5 + Coherent(0, x=-1) / 2**0.5
            >>> # state.visualize_2d()

        Args:
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``\'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.
            min_shape: The minimum fock shape to use for the Wigner function plot (default 50).

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

    def visualize_3d(
        self,
        xbounds: tuple[int] = (-6, 6),
        pbounds: tuple[int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "RdBu",
        return_fig: bool = False,
        min_shape: int = 50,
    ) -> go.Figure | None:
        r"""
        3D visualization of the Wigner function of this state on a surface plot.

        Args:
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``\'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.
            min_shape: The minimum fock shape to use for the Wigner function plot (default 50).

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
        r"""
        Plots the absolute value :math:`abs(\rho)` of the density matrix :math:`\rho` of this state
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

    @property
    def wigner(self):
        r"""
        Returns the Wigner function of this state in phase space as an ``Ansatz``.

        .. code-block::

            >>> import numpy as np
            >>> from mrmustard.lab import Ket

            >>> state = Ket.random([0])
            >>> x = np.linspace(-5, 5, 100)

            >>> assert np.all(state.wigner(x,0).real >= 0)
        """
        if isinstance(self.ansatz, PolyExpAnsatz):
            return (self >> BtoPS(self.modes, s=0)).ansatz.PS
        raise ValueError(
            "Wigner ansatz not implemented for Fock states. Consider calling ``.to_bargmann()`` first.",
        )
