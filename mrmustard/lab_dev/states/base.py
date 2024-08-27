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

from typing import Sequence

from enum import Enum
import warnings

import numpy as np
from IPython.display import display
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mrmustard import math, settings, widgets
from mrmustard.physics.fock import quadrature_distribution
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    RealVector,
    Scalar,
)
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
)
from mrmustard.math.lattice.strategies.vanilla import autoshape_numba
from mrmustard.physics.gaussian import purity
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.lab_dev.utils import shape_check
from mrmustard.physics.ansatze import (
    bargmann_Abc_to_phasespace_cov_means,
)
from mrmustard.lab_dev.circuit_components_utils import BtoPS, BtoQ, TraceOut
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.wires import Wires

__all__ = ["State", "DM", "Ket"]

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
    def probability(self) -> float:
        r"""
        Returns :math:`\langle\psi|\psi\rangle` for ``Ket`` states
        :math:`|\psi\rangle` and :math:`\text{Tr}(\rho)` for ``DM`` states :math:`\rho`.
        """
        raise NotImplementedError

    @property
    def purity(self) -> float:
        r"""
        The purity of this state.
        """
        raise NotImplementedError

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
            >>> from mrmustard.lab_dev.states.base import Ket

            >>> modes = [0, 1]
            >>> triple = coherent_state_Abc(x=[0.1, 0.2])  # parallel coherent states

            >>> coh = Ket.from_bargmann(modes, triple)
            >>> assert coh.modes == modes
            >>> assert coh.representation == Bargmann(*triple)
            >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A state.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.
        """
        return cls(modes, Bargmann(*triple), name)

    @classmethod
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
            modes: The modes of this states.
            array: The Fock array.
            name: The name of this state.
            batched: Whether the given array is batched.

        Returns:
            A state.

        Raises:
            ValueError: If the given array has a shape that is inconsistent with the number of
                modes.
        """
        return cls(modes, Fock(array, batched), name)

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        means: ComplexMatrix,
        name: str | None = None,
        atol_purity: float | None = 1e-5,
    ) -> Ket | DM:  # pylint: disable=abstract-method
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
        raise NotImplementedError

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
        Q = cls(modes, Bargmann(*triple))
        return cls(modes, (Q >> QtoB).representation, name)

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
        return bargmann_Abc_to_phasespace_cov_means(*new_state.bargmann_triple(batched=True))

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
        state = state if isinstance(state, DM) else state.dm()
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
        state = state if isinstance(state, DM) else state.dm()
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
        state = state if isinstance(state, DM) else state.dm()
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

    def _ipython_display_(self):  # pragma: no cover
        is_ket = isinstance(self, Ket)
        is_fock = isinstance(self.representation, Fock)
        display(widgets.state(self, is_ket=is_ket, is_fock=is_fock))


class DM(State):
    r"""
    Base class for density matrices.

    Args:
        modes: The modes of this density matrix.
        representation: The representation of this density matrix.
        name: The name of this density matrix.
    """

    short_name = "DM"

    def __init__(
        self,
        modes: Sequence[int] = (),
        representation: Bargmann | Fock | None = None,
        name: str | None = None,
    ):
        if representation and representation.ansatz.num_vars != 2 * len(modes):
            raise ValueError(
                f"Expected a representation with {2*len(modes)} variables, found {representation.ansatz.num_vars}."
            )
        super().__init__(
            wires=[modes, (), modes, ()],
            name=name,
        )
        if representation is not None:
            self._representation = representation

    @property
    def is_positive(self) -> bool:
        r"""
        Whether this DM is a positive operator.
        """
        batch_dim = self.representation.ansatz.batch_size
        if batch_dim > 1:
            raise ValueError(
                "Physicality conditions are not implemented for batch dimension larger than 1."
            )
        A = self.representation.A[0]
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
        r"""Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states."""
        return math.sum(self._probabilities)

    @property
    def purity(self) -> float:
        return self.L2_norm

    @property
    def _probabilities(self) -> RealVector:
        r"""Element-wise probabilities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states.
        """
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices
        rep = self.representation.trace(idx_ket, idx_bra)
        return math.real(math.sum(rep.scalar))

    @property
    def _purities(self) -> RealVector:
        r"""Element-wise purities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states.
        """
        return self._L2_norms / self._probabilities

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
        return coeff * DM(
            modes,
            Bargmann.from_function(fn=wigner_to_bargmann_rho, cov=cov, means=means),
            name,
        )

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
        full_wires = list(modes) + ancilla

        psi = Ket.random(full_wires, max_r)
        return psi[modes]

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
        if self.representation.ansatz.batch_size == 1:
            try:  # fock
                shape = self._representation.array.shape[1:]
            except AttributeError:  # bargmann
                if self.representation.ansatz.polynomial_shape[0] == 0:
                    repr = self.representation
                    A, b, c = repr.A[0], repr.b[0], repr.c[0]
                    repr = repr / self.probability
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

    def normalize(self) -> DM:
        r"""
        Returns a rescaled version of the state such that its probability is 1.
        """
        return self / self.probability

    def __getitem__(self, modes: int | Sequence[int]) -> State:
        r"""
        Traces out all the modes except those given.
        The result is returned with modes in increasing order.
        """
        if isinstance(modes, int):
            modes = [modes]
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
        representation = self.representation.trace(idxz, idxz_conj)

        return self.__class__._from_attributes(
            representation, wires, self.name
        )  # pylint: disable=protected-access

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

        w = result.wires
        if not w.input and w.bra.modes == w.ket.modes:
            return DM(w.modes, result.representation)
        return result


class Ket(State):
    r"""
    Base class for all Hilbert space vectors.

    Arguments:
        modes: The modes of this ket.
        representation: The representation of this ket.
        name: The name of this ket.
    """

    short_name = "Ket"

    def __init__(
        self,
        modes: Sequence[int] = (),
        representation: Bargmann | Fock | None = None,
        name: str | None = None,
    ):
        if representation and representation.ansatz.num_vars != len(modes):
            raise ValueError(
                f"Expected a representation with {len(modes)} variables, found {representation.ansatz.num_vars}."
            )
        super().__init__(
            wires=[(), (), modes, ()],
            name=name,
        )
        if representation is not None:
            self._representation = representation

    @property
    def is_physical(self) -> bool:
        r"""
        Whether the ket object is a physical one.
        """
        batch_dim = self.representation.ansatz.batch_size
        if batch_dim > 1:
            raise ValueError(
                "Physicality conditions are not implemented for batch dimension larger than 1."
            )

        A = self.representation.A[0]

        return all(math.abs(math.eigvals(A)) < 1) and math.allclose(
            self.probability, 1, settings.ATOL
        )

    @property
    def probability(self) -> float:
        r"""Probability of this Ket (L2 norm squared)."""
        return self.L2_norm

    @property
    def purity(self) -> float:
        return 1.0

    @property
    def _probabilities(self) -> RealVector:
        r"""Element-wise L2 norm squared along the batch dimension of this Ket."""
        return self._L2_norms

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        triple: tuple,
        name: str | None = None,
        atol_purity: float | None = 1e-5,
    ) -> Ket:
        cov, means, coeff = triple
        cov = math.astensor(cov)
        means = math.astensor(means)
        shape_check(cov, means, 2 * len(modes), "Phase space")
        if atol_purity:
            p = purity(cov)
            if p < 1.0 - atol_purity:
                msg = f"Cannot initialize a Ket: purity is {p:.5f} (must be at least 1.0-{atol_purity})."
                raise ValueError(msg)
        return Ket(
            modes,
            coeff * Bargmann.from_function(fn=wigner_to_bargmann_psi, cov=cov, means=means),
            name,
        )

    @classmethod
    def random(cls, modes: Sequence[int], max_r: float = 1.0) -> Ket:
        r"""
        Generates a random zero displacement state.

        Args:
            modes: The modes of the state.
            max_r: Maximum squeezing parameter over which we make random choices.
        Output is a Ket
        """

        m = len(modes)
        S = math.random_symplectic(m, max_r)
        transformation = (
            1
            / np.sqrt(2)
            * math.block(
                [
                    [
                        math.eye(m, dtype=math.complex128),
                        math.eye(m, dtype=math.complex128),
                    ],
                    [
                        -1j * math.eye(m, dtype=math.complex128),
                        1j * math.eye(m, dtype=math.complex128),
                    ],
                ]
            )
        )
        S = math.conj(math.transpose(transformation)) @ S @ transformation
        S_1 = S[:m, :m]
        S_2 = S[:m, m:]
        A = S_2 @ math.conj(math.inv(S_1))  # use solve for inverse
        b = math.zeros(m, dtype=A.dtype)
        psi = cls.from_bargmann(modes, [[A], [b], [complex(1)]])
        return psi.normalize()

    def auto_shape(
        self, max_prob=None, max_shape=None, respect_manual_shape=True
    ) -> tuple[int, ...]:
        r"""
        A good enough estimate of the Fock shape of this Ket, defined as the shape of the Fock
        array (batch excluded) if it exists, and if it doesn't exist it is computed as the shape
        that captures at least ``settings.AUTOSHAPE_PROBABILITY`` of the probability mass of each
        single-mode marginal (default 99.9%).
        If the ``respect_manual_shape`` flag is set to ``True``, auto_shape will respect the
        non-None values in ``manual_shape``.

        Args:
            max_prob: The maximum probability mass to capture in the shape (default from ``settings.AUTOSHAPE_PROBABILITY``).
            max_shape: The maximum shape to compute (default from ``settings.AUTOSHAPE_MAX``).
            respect_manual_shape: Whether to respect the non-None values in ``manual_shape``.
        """
        # experimental:
        if self.representation.ansatz.batch_size == 1:
            try:  # fock
                shape = self._representation.array.shape[1:]
            except AttributeError:  # bargmann
                if self.representation.ansatz.polynomial_shape[0] == 0:
                    repr = self.representation.conj() & self.representation
                    A, b, c = repr.A[0], repr.b[0], repr.c[0]
                    repr = repr / self.probability
                    shape = autoshape_numba(
                        math.asnumpy(A),
                        math.asnumpy(b),
                        math.asnumpy(c),
                        max_prob or settings.AUTOSHAPE_PROBABILITY,
                        max_shape or settings.AUTOSHAPE_MAX,
                    )
                else:
                    shape = [settings.AUTOSHAPE_MAX] * len(self.modes)
        else:
            warnings.warn("auto_shape only looks at the shape of the first element of the batch.")
            shape = [settings.AUTOSHAPE_MAX] * len(self.modes)
        if respect_manual_shape:
            return tuple(c or s for c, s in zip(self.manual_shape, shape))
        return tuple(shape)

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``Ket``.
        """
        dm = self @ self.adjoint
        ret = DM._from_attributes(dm.representation, dm.wires, self.name)
        ret.manual_shape = self.manual_shape + self.manual_shape
        return ret

    def expectation(self, operator: CircuitComponent):
        r"""
        The expectation value of an operator calculated with respect to this Ket.

        Given the operator `O`, this function returns :math:`Tr\big(|\psi\rangle\langle\psi| O)`\,
        where :math:`|\psi\rangle` is the vector representing this state.

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
            result = self @ operator.dual
            result @= result.adjoint
            result >>= TraceOut(leftover_modes)

        elif op_type is OperatorType.DM_LIKE:
            result = (self.adjoint @ (self @ operator.dual)) >> TraceOut(leftover_modes)

        else:
            result = (self @ operator) >> self.dual

        return result

    def normalize(self) -> Ket:
        r"""
        Returns a rescaled version of the state such that its probability is 1
        """
        return self / math.sqrt(self.probability)

    def __getitem__(self, modes: int | Sequence[int]) -> State:
        r"""
        Reduced density matrix obtained by tracing out all the modes except those in the given
        ``modes``. Note that the result is returned with modes in increasing order.
        """
        if isinstance(modes, int):
            modes = [modes]
        modes = set(modes)

        if not modes.issubset(self.modes):
            raise ValueError(f"Expected a subset of `{self.modes}, found `{list(modes)}`.")

        if self._parameter_set:
            # if ``self`` has a parameter set, it is a built-in state, and we slice the
            # parameters
            return self._getitem_builtin(modes)

        # if ``self`` has no parameter set, it is not a built-in state.
        # we must turn it into a density matrix and slice the representation
        return self.dm()[modes]

    def __rshift__(self, other: CircuitComponent | Scalar) -> CircuitComponent | Batch[Scalar]:
        r"""
        Contracts ``self`` and ``other`` (output of self into the inputs of other),
        adding the adjoints when they are missing. Given this is a ``Ket`` object which
        has only ket wires at the output, in expressions like ``ket >> channel`` where ``channel``
        has wires on the ket and bra sides the adjoint of ket is automatically added, effectively
        calling ``ket.adjoint @ (ket @ channel)`` and the method returns a new ``DM``.
        In expressions lke ``ket >> u`` where ``u`` is a unitary, the adjoint of ``ket`` is
        not needed and the method returns a new ``Ket``.

        Returns a ``DM`` or a ``Ket`` when the wires of the resulting components are compatible
        with those of a ``DM`` or of a ``Ket``. Returns a ``CircuitComponent`` in general,
        and a (batched) scalar if there are no wires left, for convenience.
        """
        result = super().__rshift__(other)
        if not isinstance(result, CircuitComponent):
            return result  # scalar case handled here

        if not result.wires.input:
            if not result.wires.bra:
                return Ket(result.wires.modes, result.representation)
            elif result.wires.bra.modes == result.wires.ket.modes:
                result = DM(result.wires.modes, result.representation)
        return result
