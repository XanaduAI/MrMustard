# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module containing utility classes and functions for graphical display."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from mrmustard import settings
from mrmustard.physics.fock import quadrature_distribution
from mrmustard.physics.wigner import wigner_discretized


# pylint: disable=disallowed-name
class Progressbar:
    "A spiffy loading bar to display the progress during an optimization."

    def __init__(self, max_steps: int):
        self.taskID = None
        if max_steps == 0:
            self.bar = Progress(
                TextColumn("Step {task.completed}/∞"),
                BarColumn(),
                TextColumn("Cost = {task.fields[loss]:.5f}"),
            )
        else:
            self.bar = Progress(
                TextColumn("Step {task.completed}/{task.total} | {task.fields[speed]:.1f} it/s"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Cost = {task.fields[loss]:.5f} | ⏳ "),
                TimeRemainingColumn(),
            )
        self.taskID = self.bar.add_task(
            description="Optimizing...",
            start=max_steps > 0,
            speed=0.0,
            total=max_steps,
            loss=1.0,
            refresh=True,
            visible=settings.PROGRESSBAR,
        )

    def step(self, loss):
        """Update bar step and the loss information associated with it."""
        speed = self.bar.tasks[0].speed or 0.0
        self.bar.update(self.taskID, advance=1, refresh=True, speed=speed, loss=loss)

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.bar.__exit__(exc_type, exc_val, exc_tb)


def mikkel_plot(
    rho: np.ndarray,
    xbounds: Tuple[int] = (-6, 6),
    ybounds: Tuple[int] = (-6, 6),
    **kwargs,
):  # pylint: disable=too-many-statements
    """Plots the Wigner function of a state given its density matrix.

    Args:
        rho (np.ndarray): density matrix of the state
        xbounds (Tuple[int]): range of the x axis
        ybounds (Tuple[int]): range of the y axis

    Keyword args:
        resolution (int): number of points used to calculate the wigner function
        xticks (Tuple[int]): ticks of the x axis
        xtick_labels (Optional[Tuple[str]]): labels of the x axis; if None uses default formatter
        yticks (Tuple[int]): ticks of the y axis
        ytick_labels (Optional[Tuple[str]]): labels of the y axis; if None uses default formatter
        grid (bool): whether to display the grid
        cmap (matplotlib.colormap): colormap of the figure

    Returns:
        tuple: figure and axes
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

    q, ProbX = quadrature_distribution(rho)
    p, ProbP = quadrature_distribution(rho, np.pi / 2)

    xvec = np.linspace(*xbounds, plot_args["resolution"])
    pvec = np.linspace(*ybounds, plot_args["resolution"])
    W, X, P = wigner_discretized(rho, xvec, pvec, settings.HBAR)

    ### PLOTTING ###

    fig, ax = plt.subplots(
        2, 2, figsize=(6, 6), gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]}
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Wigner function
    ax[1][0].contourf(X, P, W, 120, cmap=plot_args["cmap"], vmin=-abs(W).max(), vmax=abs(W).max())
    ax[1][0].set_xlabel("$x$", fontsize=12)
    ax[1][0].set_ylabel("$p$", fontsize=12)
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
    ax[0][0].set_ylabel("Prob($x$)", fontsize=12)
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
    ax[1][1].set_xlabel("Prob($p$)", fontsize=12)
    ax[1][1].set_xlim([0, 1.1 * max(ProbP)])
    ax[1][1].set_ylim(ybounds)
    ax[1][1].grid(plot_args["grid"])

    # Density matrix
    ax[0][1].matshow(abs(rho), cmap=plot_args["cmap"], vmin=-abs(rho).max(), vmax=abs(rho).max())
    ax[0][1].set_title(r"abs($\rho$)", fontsize=12)
    ax[0][1].tick_params(direction="in")
    ax[0][1].get_xaxis().set_ticks([])
    ax[0][1].get_yaxis().set_ticks([])
    ax[0][1].set_aspect("auto")
    ax[0][1].set_ylabel(f"cutoff = {len(rho)}", fontsize=12)
    ax[0][1].yaxis.set_label_position("right")

    return fig, ax


def wave_function_polar(x, fx):
    r"""
    Plots the complex values of a function f(x) for each value of x on the y-z plane at x,
    in polar coordinates. The points are colored by their phase angle using HUE colors.
    :param f: A function that takes a complex number as input and returns a complex number.
    :param x_range: A vector of values along the x axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    r = np.abs(fx)
    theta = np.angle(fx)
    hue = (theta + np.pi) / (2 * np.pi)  # map phase angle to hue (0 to 1)
    colors = cm.hsv(hue)

    ax.scatter(x, r * np.cos(theta), r * np.sin(theta), c=colors, marker="o", alpha=1)

    ax.set_xlabel("x")
    ax.set_ylabel("Re(f)")
    ax.set_zlabel("Im(f)")
    ax.set_facecolor("white")
    plt.rcParams["grid.color"] = (0.5, 0.5, 0.5, 0.3)
    ax.grid(True)

    # Center the Real(f(x)) and Imag(f(x)) axes at 0
    y_max = np.max(np.abs(np.real(fx)))
    z_max = np.max(np.abs(np.imag(fx)))
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(-z_max, z_max)

    plt.show()
    return ax


def wave_function_cartesian(x, fx):
    phase = np.angle(fx)
    magnitude = np.abs(fx)
    hue = (phase + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots()
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = magnitude[i], magnitude[i + 1]
        # Fill the area under the curve with a color based on the phase angle
        ax.fill_between(
            [x0, x1],
            [y0, y1],
            color=cm.hsv(hue[i]),
            alpha=0.5,
        )
    # Plot the curve in black color with linewidth=1
    ax.plot(x, magnitude, color="black", linewidth=1)
    return ax
