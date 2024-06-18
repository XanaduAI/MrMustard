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

from typing import Optional, Sequence, Union
import os

from enum import Enum
from IPython.display import display, HTML
from mako.template import Template
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from mrmustard import math, settings
from mrmustard.math.parameters import Variable
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
from mrmustard.physics.bargmann import wigner_to_bargmann_psi, wigner_to_bargmann_rho
from mrmustard.physics.converters import to_fock
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

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
    ) -> State:
        r"""
        Initializes a state of type ``cls`` from an ``(A, b, c)`` triple
        parametrizing the Ansatz in Bargmann representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Bargmann
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Ket

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
        name: Optional[str] = None,
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
            >>> assert coh.representation == Fock(array)
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
        name: Optional[str] = None,
        atol_purity: Optional[float] = 1e-5,
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
        name: Optional[str] = None,
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
    def is_pure(self):
        r"""
        Whether this state is pure.
        """
        return math.allclose(self.purity, 1.0)

    def fock(self, shape: Optional[Union[int, Sequence[int]]] = None) -> ComplexTensor:
        r"""
        The array that describes this state in the Fock representation.

        Uses the :meth:`mrmustard.physics.converters.to_fock` method to convert the internal
        representation into a ``Fock`` object.

        Args:
            shape: The shape of the returned array. If ``shape``is given as an ``int``, it is
            broadcasted to all the dimensions. If ``None``, it defaults to the value of
            ``AUTOCUTOFF_MAX_CUTOFF`` in the settings.

        Returns:
            The array that describes this state in the Fock representation.
        """
        return to_fock(self.representation, shape).array

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
        return bargmann_Abc_to_phasespace_cov_means(*new_state.representation.triple)

    def visualize_2d(
        self,
        xbounds: tuple[int, int] = (-6, 6),
        pbounds: tuple[int, int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "viridis",
        return_fig: bool = False,
    ) -> Union[go.Figure, None]:
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

        Returns:
            A ``Plotly`` figure representing the state in 2D.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes > 1:
            raise ValueError("2D visualization not available for multi-mode states.")

        state = self.to_fock(settings.AUTOCUTOFF_MAX_CUTOFF)
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
            column_widths=[2, 1],
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
            colorscale=colorscale,
            name="Wigner function",
        )
        fig.add_trace(fig_21, row=2, col=1)
        fig.update_traces(row=2, col=1, showscale=False)
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
            width=500,
            plot_bgcolor="aliceblue",
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
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
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")  # pragma: no cover
        display(HTML(html))  # pragma: no cover

    def visualize_3d(
        self,
        xbounds: tuple[int] = (-6, 6),
        pbounds: tuple[int] = (-6, 6),
        resolution: int = 200,
        colorscale: str = "viridis",
        return_fig: bool = False,
    ) -> Union[go.Figure, None]:
        r"""
        3D visualization of the Wigner function of this state on a surface plot.

        Args:
            xbounds: The range of the `x` axis.
            pbounds: The range of the `p` axis.
            resolution: The number of bins on each axes.
            colorscale: A colorscale. Must be one of ``Plotly``\'s built-in continuous color
                scales.
            return_fig: Whether to return the ``Plotly`` figure.

        Returns:
            A ``Plotly`` figure representing the state in 3D.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes != 1:
            raise ValueError("3D visualization not available for multi-mode states.")

        state = self.to_fock(settings.AUTOCUTOFF_MAX_CUTOFF)
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
                colorscale=colorscale,
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
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")  # pragma: no cover
        display(HTML(html))  # pragma: no cover

    def visualize_dm(
        self,
        cutoff: Optional[int] = None,
        return_fig: bool = False,
    ) -> Union[go.Figure, None]:
        r"""
        Plots the absolute value :math:`abs(\rho)` of the density matrix :math:`\rho` of this state
        on a heatmap.

        Args:
            cutoff: The desired cutoff. Defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` in the
                settings.
            return_fig: Whether to return the ``Plotly`` figure.

        Returns:
            A ``Plotly`` figure representing absolute value of the density matrix of this state.

        Raises:
            ValueError: If this state is a multi-mode state.
        """
        if self.n_modes != 1:
            raise ValueError("DM visualization not available for multi-mode states.")
        state = self.to_fock(cutoff or settings.AUTOCUTOFF_MAX_CUTOFF)
        state = state if isinstance(state, DM) else state.dm()
        dm = math.sum(state.representation.array, axes=[0])

        fig = go.Figure(
            data=go.Heatmap(z=abs(dm), colorscale="viridis", name="abs(ρ)", showscale=False)
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=257,
            width=257,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig.update_xaxes(title_text=f"abs(ρ), cutoff={dm.shape[0]}")

        if return_fig:
            return fig
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")  # pragma: no cover
        display(HTML(html))  # pragma: no cover

    def _repr_html_(self):  # pragma: no cover
        template = Template(filename=os.path.dirname(__file__) + "/assets/states.txt")
        display(HTML(template.render(state=self)))

    def _getitem_builtin_state(self, modes: set[int]):
        r"""
        A convenience function to slice built-in states.

        Built-in states come with a parameter set. To slice them, we simply slice the parameter
        set, and then used the sliced parameter set to re-initialize them.

        This approach avoids computing the representation, which may be expensive. Additionally,
        it allows returning trainable states.
        """
        # slice the parameter set
        items = [i for i, m in enumerate(self.modes) if m in modes]
        kwargs = {}
        for name, param in self._parameter_set[items].all_parameters.items():
            kwargs[name] = param.value
            if isinstance(param, Variable):
                kwargs[name + "_trainable"] = True
                kwargs[name + "_bounds"] = param.bounds

        return self.__class__(modes, **kwargs)


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
        modes: Sequence[int, ...] | CircuitComponent = (),
        representation: Optional[Bargmann | Fock] = None,
        name: Optional[str] = None,
    ):
        if isinstance(modes, CircuitComponent):
            cc = modes
            if not cc.wires.dm_like:
                raise ValueError("Expected a density matrix-like circuit component.")
            return DM._from_attributes(cc.representation, cc.wires)

        if representation and representation.ansatz.num_vars != 2 * len(modes):
            raise ValueError(
                f"Expected a representation with {2*len(modes)} variables, found {representation.ansatz.num_vars}."
            )
        super().__init__(
            modes_out_bra=modes,
            modes_out_ket=modes,
            name=name,
        )
        if representation is not None:
            self._representation = representation

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
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
        return coeff * DM(modes, Bargmann(*wigner_to_bargmann_rho(cov, means)), name)

    @property
    def _probabilities(self) -> RealVector:
        r"""Element-wise probabilities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states."""
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices
        rep = self.representation.trace(idx_ket, idx_bra)
        return math.real(math.sum(rep.scalar))

    @property
    def probability(self) -> float:
        r"""Probability (trace) of this DM, using the batch dimension of the Ansatz
        as a convex combination of states."""
        return math.sum(self._probabilities)

    @property
    def _purities(self) -> RealVector:
        r"""Element-wise purities along the batch dimension of this DM.
        Useful for cases where the batch dimension does not mean a convex combination of states."""
        return self._L2_norms / self._probabilities

    @property
    def purity(self) -> float:
        return self.L2_norm

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

    def __getitem__(self, modes: Union[int, Sequence[int]]) -> State:
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
            return self._getitem_builtin_state(modes)

        # if ``self`` has no parameter set it is not a built-in state,
        # in which case we trace the representation
        wires = Wires(modes_out_bra=modes, modes_out_ket=modes)

        idxz = [i for i, m in enumerate(self.modes) if m not in modes]
        idxz_conj = [i + len(self.modes) for i, m in enumerate(self.modes) if m not in modes]
        representation = self.representation.trace(idxz, idxz_conj)

        return self.__class__._from_attributes(
            representation, wires, self.name
        )  # pylint: disable=protected-access


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
        modes: tuple[int, ...] | CircuitComponent = (),
        representation: Optional[Bargmann | Fock] = None,
        name: Optional[str] = None,
    ):
        if isinstance(modes, CircuitComponent):
            cc = modes
            if not cc.wires.ket_like:
                raise ValueError("Expected a ket-like circuit component.")
            return Ket._from_attributes(cc.representation, cc.wires)

        if representation and representation.ansatz.num_vars != len(modes):
            raise ValueError(
                f"Expected a representation with {len(modes)} variables, found {representation.ansatz.num_vars}."
            )
        super().__init__(
            modes_out_ket=modes,
            name=name,
        )
        if representation is not None:
            self._representation = representation

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        triple: tuple,
        name: Optional[str] = None,
        atol_purity: Optional[float] = 1e-5,
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
        return Ket(modes, coeff * Bargmann(*wigner_to_bargmann_psi(cov, means)), name)

    @property
    def _probabilities(self) -> RealVector:
        r"""Element-wise L2 norm squared along the batch dimension of this Ket."""
        return self._L2_norms

    @property
    def probability(self) -> float:
        r"""Probability of this Ket (L2 norm squared)."""
        return self.L2_norm

    @property
    def _purities(self) -> float:
        r"""Purity of each ket in the batch."""
        return math.ones((self.representation.ansatz.batch_size,), math.float64)

    @property
    def purity(self) -> float:
        return 1.0

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``Ket``.
        """
        dm = self @ self.adjoint
        return DM._from_attributes(dm.representation, dm.wires, self.name)

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
            result = (self @ operator.dual) >> TraceOut(leftover_modes)
            result = math.abs(result) ** 2 if not leftover_modes else result

        elif op_type is OperatorType.DM_LIKE:
            result = (self.adjoint @ (self @ operator.dual)) >> TraceOut(leftover_modes)

        else:
            result = (self @ operator) >> self.dual

        return result

    def __getitem__(self, modes: Union[int, Sequence[int]]) -> State:
        r"""
        Reduced density matrix obtained by tracing out all the modes except those in the given
        ``modes``. Note that the result is returned with modes in increasing order.
        """
        if isinstance(modes, int):
            modes = [modes]
        modes = set(modes)

        if not modes.issubset(self.modes):
            msg = f"Expected a subset of `{self.modes}, found `{list(modes)}`."
            raise ValueError(msg)

        if self._parameter_set:
            # if ``self`` has a parameter set, it is a built-in state, and we slice the
            # parameters
            return self._getitem_builtin_state(modes)

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
