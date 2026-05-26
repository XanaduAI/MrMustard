# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive Wigner function explorer for single-mode quantum states.

Provides :func:`wigner_explore`, a thin wrapper around :func:`manipulate` that
takes a function returning a single-mode ``Ket`` or ``DM``, computes the Wigner
function via :func:`wigner_discretized`, and renders it with matplotlib styling
that matches :meth:`State.visualize_2d`.

When ``stellar_roots=True``, the stellar roots of the Bargmann polynomial are
plotted alongside the Wigner heatmap in the complex plane.

Requires the ipympl backend (``%matplotlib widget``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from mrmustard.lab.states.base import State

from mrmustard.physics.stellar import plot_stellar_roots
from mrmustard.physics.stellar import stellar_roots as stellar_roots_from_fock
from mrmustard.physics.wigner import wigner_discretized

from .manipulate import ComplexSlider, manipulate

__all__ = ["wigner_explore"]

# Styling constants matching State.visualize_2d
_MARGINAL_COLOR = "steelblue"
_MARGINAL_LW = 1.5
_BG_COLOR = "aliceblue"
_SPINE_COLOR = "black"
_SPINE_WIDTH = 1
_TICK_SIZE = 9
_LABEL_FONT = "Arial Black"

# Default figure sizes (inches)
_FIGSIZE_WITH_MARGINALS = (7.0, 6.2)
_FIGSIZE_NO_MARGINALS = (5.8, 5.0)
_FIGSIZE_MARGINALS_AND_STELLAR = (13.0, 6.2)
_FIGSIZE_STELLAR_ONLY = (11.5, 5.0)


def _style_axis(ax: Axes) -> None:
    """Apply visualize_2d-matching styling to a matplotlib axis."""
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(labelsize=_TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(_SPINE_WIDTH)
        spine.set_color(_SPINE_COLOR)
        spine.set_visible(True)


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def _build_wigner_figure(
    marginals: bool,
    figsize: tuple[float, float] | None,
    stellar_roots: bool = False,
) -> tuple:
    """Create the matplotlib figure and axes for the Wigner explorer.

    Returns:
        ``(fig, ax_main, ax_top, ax_right, ax_stellar)``.  Axes that are
        not needed for the chosen layout are ``None``.
    """
    if figsize is None:
        if marginals and stellar_roots:
            figsize = _FIGSIZE_MARGINALS_AND_STELLAR
        elif marginals:
            figsize = _FIGSIZE_WITH_MARGINALS
        elif stellar_roots:
            figsize = _FIGSIZE_STELLAR_ONLY
        else:
            figsize = _FIGSIZE_NO_MARGINALS

    ax_top = ax_right = ax_stellar = None

    if marginals and stellar_roots:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            2,
            3,
            width_ratios=[4, 1, 4],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.1,
            figure=fig,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_stellar = fig.add_subplot(gs[1, 2])
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
    elif marginals:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            2,
            2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
            figure=fig,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
    elif stellar_roots:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            1,
            2,
            width_ratios=[1, 1],
            wspace=0.15,
            figure=fig,
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_stellar = fig.add_subplot(gs[0, 1])
    else:
        fig, ax_main = plt.subplots(figsize=figsize)

    return fig, ax_main, ax_top, ax_right, ax_stellar


def _state_to_wigner(
    state: State,
    min_cutoff: int,
    xvec: np.ndarray,
    pvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a quantum state to its discretized Wigner function.

    Validates that the state is single-mode, converts to a Fock-space
    density matrix, and computes the Wigner function.

    Args:
        state: A single-mode ``Ket`` or ``DM``.
        min_cutoff: Minimum Fock-space dimension.  The actual cutoff is
            ``max(min_cutoff, auto_shape)``.
        xvec: Discretized *x* quadrature values.
        pvec: Discretized *p* quadrature values.

    Returns:
        Tuple ``(W, Q, P)`` of arrays.

    Raises:
        ValueError: If *state* has more than one mode.
    """
    if state.n_modes != 1:
        raise ValueError(f"wigner_explore requires a single-mode state, got {state.n_modes} modes.")
    shape = tuple(max(min_cutoff, d) for d in state.auto_shape())
    dm = np.squeeze(state.to_fock(shape).dm().ansatz.array)
    return wigner_discretized(dm, xvec, pvec)


def _compute_stellar_roots(state: State, min_cutoff: int) -> np.ndarray:
    """Compute stellar roots from a single-mode quantum state.

    For ``Ket`` states, delegates to :meth:`Ket.stellar_roots`.  If the
    state is a linear superposition (which ``stellar_roots`` does not
    support directly), falls back to converting to Fock representation
    and computing roots from the amplitudes.

    Returns an empty array for ``DM`` states or any state without a
    ``stellar_roots`` method.
    """
    if not hasattr(state, "stellar_roots"):
        return np.array([], dtype=np.complex128)
    try:
        return state.stellar_roots()
    except ValueError:
        shape = tuple(max(min_cutoff, d) for d in state.auto_shape())
        amplitudes = np.squeeze(np.asarray(state.to_fock(shape).ansatz.array))
        return stellar_roots_from_fock(amplitudes)


def _draw_wigner_frame(
    W: np.ndarray,
    Q: np.ndarray,
    P: np.ndarray,
    *,
    ax_main: Axes,
    ax_top: Axes | None,
    ax_right: Axes | None,
    xvec: np.ndarray,
    pvec: np.ndarray,
    dx: float,
    dp: float,
    xbounds: tuple[float, float],
    pbounds: tuple[float, float],
    cmap: str,
) -> None:
    """Draw a single Wigner function frame onto the provided axes.

    Renders the Wigner heatmap on *ax_main*.  When *ax_top* and *ax_right*
    are not ``None``, also draws the *x* and *p* quadrature marginals.

    The Wigner array uses the ``np.outer(q_vec, ones)`` convention from
    :func:`wigner_discretized`, so axis 0 is *x* and axis 1 is *p*.

    Args:
        W: 2-D Wigner function array.
        Q: 2-D *x*-coordinate meshgrid (same shape as *W*).
        P: 2-D *p*-coordinate meshgrid (same shape as *W*).
        ax_main: Matplotlib axis for the heatmap.
        ax_top: Axis for the *x*-quadrature marginal, or ``None``.
        ax_right: Axis for the *p*-quadrature marginal, or ``None``.
        xvec: 1-D *x* quadrature values.
        pvec: 1-D *p* quadrature values.
        dx: Grid spacing in *x*.
        dp: Grid spacing in *p*.
        xbounds: ``(x_min, x_max)`` for axis limits.
        pbounds: ``(p_min, p_max)`` for axis limits.
        cmap: Matplotlib colormap name.
    """
    wmax = max(abs(W.min()), abs(W.max()), 1e-10)
    ax_main.pcolormesh(Q, P, W, cmap=cmap, shading="auto", vmin=-wmax, vmax=wmax)
    ax_main.set_aspect("equal")
    ax_main.set_xlim(*xbounds)
    ax_main.set_ylim(*pbounds)
    ax_main.set_xlabel("x", fontfamily=_LABEL_FONT)
    ax_main.set_ylabel("p", fontfamily=_LABEL_FONT)
    _style_axis(ax_main)

    if ax_top is not None and ax_right is not None:
        # Top marginal must match main plot width; main is constrained by set_aspect
        ax_main.apply_aspect()
        pos_main = ax_main.get_position()
        pos_top = ax_top.get_position()
        ax_top.set_position([pos_main.x0, pos_top.y0, pos_main.width, pos_top.height])
        # x-quadrature marginal (top) — integrate W over p (axis 1)
        prob_x = W.sum(axis=1) * dp
        ax_top.plot(xvec, prob_x, color=_MARGINAL_COLOR, linewidth=_MARGINAL_LW)
        ax_top.fill_between(xvec, prob_x, alpha=0.15, color=_MARGINAL_COLOR)
        ax_top.set_xlim(*xbounds)
        ax_top.set_ylim(0, None)
        ax_top.set_ylabel("Prob(x)", fontsize=9)
        ax_top.tick_params(labelbottom=False)
        _style_axis(ax_top)

        # p-quadrature marginal (right) — integrate W over x (axis 0)
        prob_p = W.sum(axis=0) * dx
        ax_right.plot(prob_p, pvec, color=_MARGINAL_COLOR, linewidth=_MARGINAL_LW)
        ax_right.fill_betweenx(pvec, prob_p, alpha=0.15, color=_MARGINAL_COLOR)
        ax_right.set_ylim(*pbounds)
        ax_right.set_xlim(0, None)
        ax_right.set_xlabel("Prob(p)", fontsize=9)
        ax_right.tick_params(labelleft=False)
        _style_axis(ax_right)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wigner_explore(
    state_fn: Callable,
    *,
    xbounds: tuple[float, float] = (-6, 6),
    pbounds: tuple[float, float] = (-6, 6),
    resolution: int = 200,
    min_cutoff: int = 50,
    cmap: str = "RdBu",
    marginals: bool = True,
    stellar_roots: bool = False,
    continuous_update: bool = False,
    figsize: tuple[float, float] | None = None,
    **slider_specs: tuple[float, ...] | ComplexSlider,
) -> None:  # pragma: no cover
    """Interactive Wigner function explorer with sliders.

    Creates sliders for each named parameter, calls *state_fn* to obtain a
    single-mode quantum state, computes its Wigner function, and plots the
    result in a style matching :meth:`State.visualize_2d`.

    When *marginals* is ``True`` (default) the *x* and *p* quadrature
    probability distributions are shown on top of and to the right of the
    Wigner heatmap, mirroring the Plotly layout of ``visualize_2d``.

    When *stellar_roots* is ``True`` the stellar roots of the Bargmann
    polynomial are plotted in the complex plane alongside the Wigner
    heatmap.  Roots are only available for ``Ket`` states; ``DM`` states
    show an empty panel.

    Uses :func:`manipulate` in custom-figure mode.  Requires the ipympl
    backend (``%matplotlib widget``).

    Args:
        state_fn: A callable whose keyword arguments match the slider names.
            Must return a single-mode ``Ket`` or ``DM``.
        xbounds: Range of the *x* quadrature axis.
        pbounds: Range of the *p* quadrature axis.
        resolution: Number of grid points per axis (default 200, matching
            ``visualize_2d``).
        min_cutoff: Minimum Fock-space dimension used when converting the
            state to a density matrix.  The actual cutoff is
            ``max(min_cutoff, auto_shape)``.
        cmap: Matplotlib colormap name.  Default ``"RdBu"`` matches the
            ``visualize_2d`` colour palette.
        marginals: If ``True`` (default), show the *x* and *p* quadrature
            distributions alongside the Wigner heatmap.
        stellar_roots: If ``True``, show the stellar roots of the Bargmann
            polynomial in the complex plane, to the right of the Wigner
            heatmap.  Only meaningful for ``Ket`` states; ``DM`` states
            show an empty panel.
        continuous_update: If ``False``, recompute only on slider
            release.  Set to ``True`` (default) for cheap computations.
        figsize: Figure size in inches.  When ``None`` (default) a sensible
            size is chosen depending on *marginals* and *stellar_roots*.
        **slider_specs: Slider specifications forwarded to
            :func:`manipulate`.  See its docstring for the accepted formats.

    Raises:
        ValueError: If the state returned by *state_fn* has more than one
            mode.  Only single-mode Wigner functions are supported.

    Examples::

        from mrmustard.lab import Coherent
        from mrmustard.widgets import wigner_explore

        def cat_state(alpha):
            return (Coherent(0, alpha=alpha) + Coherent(0, alpha=-alpha)).normalize()

        wigner_explore(cat_state, alpha=(0.3, 3.0, 0.01, 1.0), continuous_update=True)

        # With stellar roots panel:
        wigner_explore(cat_state, alpha=(0.3, 3.0, 0.01, 1.0), continuous_update=True, stellar_roots=True)
    """
    xvec = np.linspace(*xbounds, resolution)
    pvec = np.linspace(*pbounds, resolution)
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]

    fig, ax_main, ax_top, ax_right, ax_stellar = _build_wigner_figure(
        marginals, figsize, stellar_roots
    )
    all_axes = [a for a in (ax_main, ax_top, ax_right, ax_stellar) if a is not None]

    def _plot(ax, **kwargs):
        state = state_fn(**kwargs)
        W, Q, P = _state_to_wigner(state, min_cutoff, xvec, pvec)
        _draw_wigner_frame(
            W,
            Q,
            P,
            ax_main=ax_main,
            ax_top=ax_top,
            ax_right=ax_right,
            xvec=xvec,
            pvec=pvec,
            dx=dx,
            dp=dp,
            xbounds=xbounds,
            pbounds=pbounds,
            cmap=cmap,
        )
        if ax_stellar is not None:
            roots = _compute_stellar_roots(state, min_cutoff)
            plot_stellar_roots(roots, ax=ax_stellar)

    manipulate(
        _plot,
        fig,
        all_axes,
        continuous_update=continuous_update,
        **slider_specs,
    )
