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
This module contains the functions to visualize states in the phase space.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from mrmustard.physics import fock
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import ComplexMatrix

__all__ = ["mikkel_plot"]


def mikkel_plot(
    dm: ComplexMatrix,
    xbounds: tuple[int] = (-6, 6),
    ybounds: tuple[int] = (-6, 6),
    **kwargs,
):  # pylint: disable=too-many-statements
    """Plots the Wigner function of a state given its density matrix.

    Args:
        dm: The density matrix to plot.
        xbounds: The range of the x axis.
        ybounds: The range of the y axis.

    Keyword args:
        resolution: The number of points used to calculate the wigner function.
        xticks: The ticks of the x axis.
        xtick_labels: The labels of the x axis; if None uses default formatter.
        yticks: The ticks of the y axis.
        ytick_labels: The labels of the y axis; if None uses default formatter.
        grid: Whether to display the grid.
        cmap: The colormap of the figure.

    Returns:
        The figure and axes.
    """

    plot_args = {
        "resolution": 200,
        "xticks": (-5, 0, 5),
        "xtick_labels": None,
        "yticks": (-5, 0, 5),
        "ytick_labels": None,
        "grid": False,
        "cmap": cm.RdBu,
    }
    plot_args.update(kwargs)

    if plot_args["xtick_labels"] is None:
        plot_args["xtick_labels"] = plot_args["xticks"]
    if plot_args["ytick_labels"] is None:
        plot_args["ytick_labels"] = plot_args["yticks"]

    q, ProbX = fock.quadrature_distribution(dm)
    p, ProbP = fock.quadrature_distribution(dm, np.pi / 2)

    xvec = np.linspace(*xbounds, plot_args["resolution"])
    pvec = np.linspace(*ybounds, plot_args["resolution"])
    W, X, P = wigner_discretized(dm, xvec, pvec)

    gridspec_kw = {"width_ratios": [2, 1], "height_ratios": [1, 2]}
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw=gridspec_kw)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    ax[1][0].contourf(X, P, W, 120, cmap=plot_args["cmap"], vmin=-abs(W).max(), vmax=abs(W).max())
    ax[1][0].set_xlabel("x", fontsize=12)
    ax[1][0].set_ylabel("p", fontsize=12)
    ax[1][0].get_xaxis().set_ticks(plot_args["xticks"])
    ax[1][0].xaxis.set_ticklabels(plot_args["xtick_labels"])
    ax[1][0].get_yaxis().set_ticks(plot_args["yticks"])
    ax[1][0].yaxis.set_ticklabels(plot_args["ytick_labels"], rotation="vertical", va="center")
    ax[1][0].tick_params(direction="in")
    ax[1][0].set_xlim(xbounds)
    ax[1][0].set_ylim(ybounds)
    ax[1][0].grid(plot_args["grid"])

    # X quadrature probability distribution
    ax[0][0].fill(q, ProbX, color=plot_args["cmap"](0.5))
    ax[0][0].plot(q, ProbX, color=plot_args["cmap"](0.8))
    ax[0][0].get_xaxis().set_ticks(plot_args["xticks"])
    ax[0][0].xaxis.set_ticklabels([])
    ax[0][0].get_yaxis().set_ticks([])
    ax[0][0].tick_params(direction="in")
    ax[0][0].set_ylabel("Prob(x)", fontsize=12)
    ax[0][0].set_xlim(xbounds)
    ax[0][0].set_ylim([0, 1.1 * max(ProbX)])
    ax[0][0].grid(plot_args["grid"])

    # P quadrature probability distribution
    ax[1][1].fill(ProbP, p, color=plot_args["cmap"](0.5))
    ax[1][1].plot(ProbP, p, color=plot_args["cmap"](0.8))
    ax[1][1].get_xaxis().set_ticks([])
    ax[1][1].get_yaxis().set_ticks(plot_args["yticks"])
    ax[1][1].yaxis.set_ticklabels([])
    ax[1][1].tick_params(direction="in")
    ax[1][1].set_xlabel("Prob(p)", fontsize=12)
    ax[1][1].set_xlim([0, 1.1 * max(ProbP)])
    ax[1][1].set_ylim(ybounds)
    ax[1][1].grid(plot_args["grid"])

    # Density matrix
    ax[0][1].matshow(abs(dm), cmap=plot_args["cmap"], vmin=-abs(dm).max(), vmax=abs(dm).max())
    ax[0][1].set_title("abs(ρ)", fontsize=12)
    ax[0][1].tick_params(direction="in")
    ax[0][1].get_xaxis().set_ticks([])
    ax[0][1].get_yaxis().set_ticks([])
    ax[0][1].set_aspect("auto")
    ax[0][1].set_ylabel(f"cutoff = {len(dm)}", fontsize=12)
    ax[0][1].yaxis.set_label_position("right")

    return fig, ax
