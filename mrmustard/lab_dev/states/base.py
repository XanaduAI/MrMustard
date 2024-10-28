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

# pylint: disable=abstract-method, chained-comparison, use-dict-literal, protected-access, inconsistent-return-statements

"""
This module contains the base classes for the available quantum states.

In the docstrings defining the available states we provide a definition in terms of
the covariance matrix :math:`V` and the vector of means :math:`r`. Additionally, we
provide the ``(A, b, c)`` triples that define the states in the Fock Bargmann
representation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from enum import Enum

import numpy as np
from IPython.display import display
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mrmustard import math, settings
from mrmustard.physics.fock import quadrature_distribution
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    RealVector,
)
from mrmustard.physics.representations import Bargmann
from mrmustard.physics.ansatze import (
    bargmann_Abc_to_phasespace_cov_means,
)
from mrmustard.lab_dev.circuit_components_utils import BtoPS
from mrmustard.lab_dev.circuit_components import CircuitComponent

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
        if not w.ket.output.modes == w.bra.output.modes:
            msg = "Found DM-like operator with different modes for ket and bra wires."
            return OperatorType.INVALID_TYPE, msg
        return OperatorType.DM_LIKE, ""

    # check if operator is unitary-like
    if w.ket.input and w.ket.output and not w.bra.input and not w.bra.input:
        if not w.ket.input.modes == w.ket.output.modes:
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
        """
        return math.sum(math.real(self >> self.dual))

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

    @property
    def _L2_norms(self) -> RealVector:
        r"""
        The `L2` norm squared of a ``Ket``, or the Hilbert-Schmidt norm of a ``DM``,
        element-wise along the batch dimension.
        """
        settings.UNSAFE_ZIP_BATCH = True
        rep = self >> self.dual
        settings.UNSAFE_ZIP_BATCH = False
        return math.real(rep)

    @classmethod
    @abstractmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: str | None = None,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` from an ``(A, b, c)`` triple
        parametrizing the Ansatz in Bargmann representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Bargmann
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev.states.ket import Ket

            >>> modes = [0, 1]
            >>> triple = coherent_state_Abc(x=[0.1, 0.2])  # parallel coherent states

            >>> coh = Ket.from_bargmann(modes, triple)
            >>> assert coh.modes == modes
            >>> assert coh.representation == Bargmann(*triple)
            >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this state.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A state.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.
        """

    @classmethod
    @abstractmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: str | None = None,
        batched: bool = False,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` from an array parametrizing the
        state in Fock representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Fock
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Coherent, Ket

            >>> modes = [0]
            >>> array = Coherent(modes, x=0.1).to_fock().representation.array
            >>> coh = Ket.from_fock(modes, array, batched=True)

            >>> assert coh.modes == modes
            >>> assert coh.representation == Fock(array, batched=True)
            >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this state.
            array: The Fock array.
            name: The name of this state.
            batched: Whether the given array is batched.

        Returns:
            A state.

        Raises:
            ValueError: If the given array has a shape that is inconsistent with the number of
                modes.
        """

    @classmethod
    @abstractmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        means: ComplexMatrix,
        name: str | None = None,
        atol_purity: float | None = 1e-5,
    ) -> State:  # pylint: disable=abstract-method
        r"""
        Initializes a state from the covariance matrix and the vector of means of a state in
        phase space.

        Note that if the given covariance matrix and vector of means are consistent with a pure
        state, a ``Ket`` is returned. Otherwise, a ``DM`` is returned. One can skip this check by
        setting ``atol_purity`` to ``None``.

        Args:
            modes: The modes of this states.
            cov: The covariance matrix.
            means: The vector of means.
            name: The name of this state.
            atol_purity: If ``atol_purity`` is given, the purity of the state is computed, and an
                error is raised if its value is smaller than ``1-atol_purity`` or larger than
                ``1+atol_purity``. If ``None``, this check is skipped.

        Returns:
            A state.

        Raises:
            ValueError: If the given ``cov`` and ``means`` have shapes that are inconsistent
                with the number of modes.
            ValueError: If ``atol_purity`` is not ``None`` and the purity of the returned state
                is smaller than ``1-atol_purity`` or larger than ``1+atol_purity``.
        """

    @classmethod
    @abstractmethod
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
        if not isinstance(self.representation, Bargmann):
            raise ValueError("Can calculate phase space only for Bargmann states.")

        new_state = self >> BtoPS(self.modes, s=s)
        return bargmann_Abc_to_phasespace_cov_means(
            *new_state.bargmann_triple(batched=True), batched=True
        )

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

            >>> from mrmustard.lab_dev import Coherent

            >>> state = Coherent([0], x=1) / 2**0.5 + Coherent([0], x=-1) / 2**0.5
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
        shape = [max(min_shape, d) for d in self.auto_shape()]
        state = self.to_fock(tuple(shape))
        state = state.dm()
        dm = math.sum(state.representation.array, axes=[0])

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
        fig_11 = go.Scatter(x=x, y=prob_x, line=dict(color="steelblue", width=2), name="Prob(x)")
        fig.add_trace(fig_11, row=1, col=1)
        fig.update_xaxes(range=xbounds, row=1, col=1, showticklabels=False)
        fig.update_yaxes(title_text="Prob(x)", range=(0, max(prob_x)), row=1, col=1)

        # P quadrature probability distribution
        fig_22 = go.Scatter(x=prob_p, y=-p, line=dict(color="steelblue", width=2), name="Prob(p)")
        fig.add_trace(fig_22, row=2, col=2)
        fig.update_xaxes(title_text="Prob(p)", range=(0, max(prob_p)), row=2, col=2)
        fig.update_yaxes(range=pbounds, row=2, col=2, showticklabels=False)

        fig.update_layout(
            height=500,
            width=580,
            plot_bgcolor="aliceblue",
            margin=dict(l=20, r=20, t=30, b=20),
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
        shape = [max(min_shape, d) for d in self.auto_shape()]
        state = self.to_fock(tuple(shape))
        state = state.dm()
        dm = math.sum(state.representation.array, axes=[0])

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
                hovertemplate="x: %{x:.3f}"
                + "<br>p: %{y:.3f}"
                + "<br>W(x, p): %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=0, r=0, b=0, t=0),
            scene_camera_eye=dict(x=-2.1, y=0.88, z=0.64),
            coloraxis={"colorscale": colorscale, "cmid": 0},
        )
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=False
            )
        )
        fig.update_traces(
            contours_y=dict(show=True, usecolormap=True, highlightcolor="red", project_y=False)
        )
        fig.update_traces(
            contours_x=dict(show=True, usecolormap=True, highlightcolor="yellow", project_x=False)
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
        state = self.to_fock(cutoff)
        state = state.dm()
        dm = math.sum(state.representation.array, axes=[0])

        fig = go.Figure(
            data=go.Heatmap(z=abs(dm), colorscale="viridis", name="abs(ρ)", showscale=False)
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=257,
            width=257,
            margin=dict(l=30, r=30, t=30, b=20),
        )
        fig.update_xaxes(title_text=f"abs(ρ), cutoff={dm.shape[0]}")

        if return_fig:
            return fig
        display(fig)
