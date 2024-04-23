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

from IPython.display import display, HTML
from mako.template import Template
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from mrmustard import math, settings
from mrmustard.math.parameters import Variable
from mrmustard.physics.fock import quadrature_distribution
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector
from mrmustard.physics.bargmann import wigner_to_bargmann_psi, wigner_to_bargmann_rho
from mrmustard.physics.converters import to_fock
from mrmustard.physics.gaussian import purity
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.physics.ansatze import (
    bargmann_Abc_to_phasespace_cov_means,
)
from ..circuit_components_utils import _DsMap, _BtoQMap
from ..circuit_components import CircuitComponent
from ..wires import Wires

__all__ = ["State", "DM", "Ket"]


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
        batched: bool = False,
    ) -> State:
        r"""
        Initializes a state from an ``(A, b, c)`` triple defining a Bargmann representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Bargmann
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Ket

            >>> modes = [0, 1]
            >>> triple = coherent_state_Abc(x=[0.1, 0.2])

            >>> coh = Ket.from_bargmann(modes, triple)
            >>> assert coh.modes == modes
            >>> assert coh.representation == Bargmann(*triple)
            >>> assert isinstance(coh, Ket)

        Args:
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.
            batched: Whether the given triple is batched.

        Returns:
            A state.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.
        """
        raise NotImplementedError

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: Optional[str] = None,
        batched: bool = False,
    ) -> State:
        r"""
        Initializes a state from an array describing the state in the Fock representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Fock
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Coherent, Ket

            >>> modes = [0]
            >>> array = Coherent(modes, x=0.1).to_fock_component().representation.array
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
        raise NotImplementedError

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        means: ComplexMatrix,
        name: Optional[str] = None,
        atol_purity: Optional[float] = 1e-3,
    ) -> State:  # pylint: disable=abstract-method
        r"""
        Initializes a state from the covariance matrix and the vector of means of a state in
        phase space.

        Args:
            cov: The covariance matrix.
            means: The vector of means.
            modes: The modes of this states.
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
        name: Optional[str] = None,
    ) -> State:
        r"""
        Initializes a state from quadrature with a ABC Ansatz Gaussian exponential form.

        Args:
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A state.

        Raises:
            ValueError: If the given triple have shapes that are inconsistent
                with the number of modes.
        """
        quadrature_rep = Bargmann.from_ansatz(*triple)  # pretend it is a Bargmann but it is not yet
        state_component = cls(name, modes)
        state_component._representation = quadrature_rep
        state_component = state_component >> _BtoQMap.dual
        state_component._representation = Bargmann(
            state_component._representation.A,
            state_component._representation.b,
            state_component._representation.c,
        )
        return state_component

    @property
    def bargmann_triple(self) -> tuple[ComplexMatrix, ComplexVector, complex]:
        r"""
        The ``(A, b, c)`` triple that describes this state in the Bargmann representation.

        Returns:
            The ``(A, b, c)`` triple that describes this state in the Bargmann representation.

        Raises:
            ValueError: If the triple cannot be calculated given the state's representation.
        """
        rep = self.representation
        if isinstance(rep, Bargmann):
            return rep.A, rep.b, rep.c
        msg = f"Cannot compute triple from representation of type ``{rep.__class__.__name__}``."
        raise ValueError(msg)

    @property
    def L2_norm(self) -> float:
        r"""
        The `L2` norm of a ``Ket``, or the Hilbert-Schmidt norm of a ``DM``.
        """
        rep = self.representation
        msg = "Method ``L2_norm`` not supported for batched representations."
        if isinstance(rep, Fock):
            if rep.array.shape[0] > 1:
                raise ValueError(msg)
        else:
            if rep.A.shape[0] > 1:
                raise ValueError(msg)

        rep = (self >> self.dual).representation
        ret = rep.c if isinstance(rep, Bargmann) else rep.array
        return math.atleast_1d(ret, math.float64)[0]

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

    def fock_array(self, shape: Optional[Union[int, Sequence[int]]] = None) -> ComplexTensor:
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

    def phase_space(self, s: float) -> tuple[ComplexMatrix, ComplexVector, complex]:
        r"""
        Returns the phase space parametrization of a state, consisting in a covariance matrix, a vector of means and a scaling coefficient. When a state is a linear superposition of Gaussians each of cov, means, coeff are arranged in a batch.
        Phase space representations are labelled by an ``s`` parameter (float) which modifies the exponent of :math:`D_s(\gamma) = e^{\frac{s}{2}|\gamma|^2}D(\gamma)`, which is the operator basis used to expand phase space density matrices.
        The ``s`` parameter typically takes the values of -1, 0, 1 to indicate Glauber/Wigner/Husimi functions. Note that the same ``(cov, means, coeff)`` triple can be used to parametrize the characteristic functions as well.

        Args:
            s: The phase space parameter

            Returns:
                The covariance matrix, the mean vector and the coefficient of the state in s-parametrized phase space.
        """
        if not isinstance(self.representation, Bargmann):
            raise ValueError(f"Can not calculate phase space for ``{self.name}`` object.")

        new_state = self >> _DsMap(self.modes, s=s)  # pylint: disable=protected-access
        return bargmann_Abc_to_phasespace_cov_means(
            new_state.representation.ansatz.A,
            new_state.representation.ansatz.b,
            new_state.representation.ansatz.c,
        )

    def visualize_2d(
        self,
        xbounds: tuple[int] = (-6, 6),
        pbounds: tuple[int] = (-6, 6),
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
        if self.n_modes != 1:
            raise ValueError("2D visualization not available for multi-mode states.")

        state = self.to_fock_component(settings.AUTOCUTOFF_MAX_CUTOFF)
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
            x=xs, y=-ps, z=math.transpose(z), colorscale=colorscale, name="Wigner function"
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

        state = self.to_fock_component(settings.AUTOCUTOFF_MAX_CUTOFF)
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
            xaxis_title_text="x", yaxis_title_text="p", zaxis_title_text="Wigner function"
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

        state = self.to_fock_component(cutoff or settings.AUTOCUTOFF_MAX_CUTOFF)
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

        # use `mro` to return the correct state
        return self.__class__(modes, **kwargs)

    def quadrature(self, modes) -> tuple[ComplexMatrix, ComplexVector, complex]:
        r"""
        The A matrix, b vector and c scalar that describe this state in the quadrature basis.
        """
        if len(list(set(modes) & set(self.modes))) != len(modes):
            raise ValueError(
                f"The modes ``{modes}`` needs to be transformed is not included in the modes ``{self.modes}`` of the state."
            )

        ret = self >> _BtoQMap(self.modes)
        message = "Quadrature rep on modes" + modes + ", Bargmann rep on other modes."
        return ret.bargmann_triple, message


class DM(State):
    r"""
    Base class for density matrices.

    Args:
        name: The name of this state.
        modes: The modes of this state.
    """

    def __init__(self, name: Optional[str] = None, modes: tuple[int, ...] = ()):
        super().__init__(
            name or "DM" + "".join(str(m) for m in sorted(modes)),
            modes_out_bra=modes,
            modes_out_ket=modes,
        )

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
        batched: bool = False,
    ) -> DM:
        A = math.astensor(triple[0])
        b = math.astensor(triple[1])
        c = math.astensor(triple[2])

        n_modes = len(modes)
        A_sh = (1, 2 * n_modes, 2 * n_modes) if batched else (2 * n_modes, 2 * n_modes)
        b_sh = (1, 2 * n_modes) if batched else (2 * n_modes,)
        if A.shape != A_sh or b.shape != b_sh:
            msg = f"Given triple is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = DM(name, modes)
        ret._representation = Bargmann(A, b, c)
        return ret

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: Optional[str] = None,
        batched: bool = False,
    ) -> DM:
        array = math.astensor(array)

        n_modes = len(modes)
        if len(array.shape) != 2 * n_modes + (1 if batched else 0):
            msg = f"Given array is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = DM(name, modes)
        ret._representation = Fock(array, batched)
        return ret

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        means: ComplexMatrix,
        name: Optional[str] = None,
        atol_purity: Optional[float] = 1e-3,
    ) -> DM:
        cov = math.astensor(cov)
        means = math.astensor(means)

        n_modes = len(modes)
        if means.shape != (2 * n_modes,):
            msg = f"Given ``means`` is inconsistent with modes=``{modes}``."
            raise ValueError(msg)
        if cov.shape != (2 * n_modes, 2 * n_modes):
            msg = f"Given ``cov`` is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        if atol_purity:
            p = purity(cov)
            if p < 1.0 - atol_purity:
                msg = f"Cannot initialize a ket: purity is {p:.3f} (must be 1.0)."
                raise ValueError(msg)

        ret = DM(name, modes)
        ret._representation = Bargmann(*wigner_to_bargmann_rho(cov, means))
        return ret

    @property
    def probability(self) -> float:
        idx_ket = self.wires.output.ket.indices
        idx_bra = self.wires.output.bra.indices

        rep = self.representation.trace(idx_ket, idx_bra)

        if isinstance(rep, Bargmann):
            return math.real(math.sum(rep.c, axes=[0]))
        return math.real(math.sum(rep.array, axes=[0]))

    @property
    def purity(self) -> float:
        return (self / self.probability).L2_norm

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``DM`` when the wires of the resulting components are compatible with those
        of a ``Ket``, a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if not ret.wires.input and ret.wires.bra.modes == ret.wires.ket.modes:
            return DM._from_attributes("", ret.representation, ret.wires)
        return ret

    def __repr__(self) -> str:
        return ""

    def __getitem__(self, modes: Union[int, Sequence[int]]) -> State:
        r"""
        Traces out all the modes, except those in the given ``modes``.
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

        # if ``self`` has no parameter set, it is not a built-in state, and we must slice the
        # representation
        wires = Wires(modes_out_bra=modes, modes_out_ket=modes)

        idxz = [i for i, m in enumerate(self.modes) if m not in modes]
        idxz_conj = [i + len(self.modes) for i, m in enumerate(self.modes) if m not in modes]
        representation = self.representation.trace(idxz, idxz_conj)

        return self.__class__._from_attributes(
            self.name, representation, wires
        )  # pylint: disable=protected-access


class Ket(State):
    r"""
    Base class for all pure states, potentially unnormalized.

    Arguments:
        name: The name of this state.
        modes: The modes of this states.
    """

    def __init__(self, name: Optional[str] = None, modes: tuple[int, ...] = ()):
        super().__init__(
            name or "Ket" + "".join(str(m) for m in sorted(modes)), modes_out_ket=modes
        )

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
        batched: bool = False,
    ) -> Ket:
        A = math.astensor(triple[0])
        b = math.astensor(triple[1])
        c = math.astensor(triple[2])

        n_modes = len(modes)
        A_sh = (1, n_modes, n_modes) if batched else (n_modes, n_modes)
        b_sh = (1, n_modes) if batched else (n_modes,)
        if A.shape != A_sh or b.shape != b_sh:
            msg = f"Given triple is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = Ket(name, modes)
        ret._representation = Bargmann(A, b, c)
        return ret

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: Optional[str] = None,
        batched: bool = False,
    ) -> Ket:
        array = math.astensor(array)

        n_modes = len(modes)
        if len(array.shape) != n_modes + (1 if batched else 0):
            msg = f"Given array is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = Ket(name, modes)
        ret._representation = Fock(array, batched)
        return ret

    @classmethod
    def from_phase_space(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        means: ComplexMatrix,
        name: Optional[str] = None,
        atol_purity: Optional[float] = 1e-3,
    ):
        cov = math.astensor(cov)
        means = math.astensor(means)

        n_modes = len(modes)
        if means.shape != (2 * n_modes,):
            msg = f"Given ``means`` is inconsistent with modes=``{modes}``."
            raise ValueError(msg)
        if cov.shape != (2 * n_modes, 2 * n_modes):
            msg = f"Given ``cov`` is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        if atol_purity:
            p = purity(cov)
            if p < 1.0 - atol_purity:
                msg = f"Cannot initialize a ket: purity is {p:.3f} (must be 1.0)."
                raise ValueError(msg)

        ret = Ket(name, modes)
        ret._representation = Bargmann(*wigner_to_bargmann_psi(cov, means))
        return ret

    @property
    def probability(self) -> float:
        rep = (self >> self.dual).representation
        if isinstance(rep, Bargmann):
            return math.real(math.sum(rep.c, axes=[0]))
        return math.real(math.sum(rep.array, axes=[0]))

    @property
    def purity(self) -> float:
        return 1.0

    def dm(self) -> DM:
        r"""
        The ``DM`` object obtained from this ``Ket``.
        """
        dm = self @ self.adjoint
        return DM._from_attributes(self.name, dm.representation, dm.wires)

    def __getitem__(self, modes: Union[int, Sequence[int]]) -> State:
        r"""
        Traces out all the modes, except those in the given ``modes``.
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

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``DM`` or a ``Ket`` when the wires of the resulting components are compatible
        with those of a ``DM`` or of a ``Ket``, a ``CircuitComponent`` otherwise.
        """
        ret = super().__rshift__(other)

        if not ret.wires.input:
            if not ret.wires.bra:
                return Ket._from_attributes("", ret.representation, ret.wires)
            if ret.wires.bra.modes == ret.wires.ket.modes:
                return DM._from_attributes("", ret.representation, ret.wires)
        return ret

    def __repr__(self) -> str:
        return ""
