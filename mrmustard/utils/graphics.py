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
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mrmustard import settings
from mrmustard.physics.fock import quadrature_distribution
from .wigner import wigner_discretized

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


def mikkel_plot(rho: np.ndarray, xbounds: Tuple[int] = (-6, 6), ybounds: Tuple[int] = (-6, 6), ticks = [-5, 0, 5], tick_labels = None, grid = False):
    """Plots the Wigner function of a state given its density matrix.

    Args:
        rho (np.ndarray): density matrix of the state
        xbounds (Tuple[int]): range of the x axis
        ybounds (Tuple[int]): range of the y axis
    """

    q, ProbX = quadrature_distribution(rho)
    p, ProbP = quadrature_distribution(rho, np.pi/2)

    xvec = np.linspace(*xbounds, 200)
    pvec = np.linspace(*ybounds, 200)
    W, X, P = wigner_discretized(rho, xvec, pvec, settings.HBAR)

    ### PLOTTING ###

    _, ax = plt.subplots(
        2, 2, figsize=(6, 6), gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]}
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Wigner function
    ax[1][0].contourf(X, P, W, 120, cmap=cm.RdBu, vmin=-abs(W).max(), vmax=abs(W).max())
    ax[1][0].set_xlabel("$x$", fontsize=12)
    ax[1][0].set_ylabel("$p$", fontsize=12)
    ax[1][0].get_xaxis().set_ticks(ticks)
    ax[1][0].xaxis.set_ticklabels(tick_labels or ticks)
    ax[1][0].get_yaxis().set_ticks(ticks)
    ax[1][0].yaxis.set_ticklabels(tick_labels or ticks)
    ax[1][0].tick_params(direction="in")
    ax[1][0].set_xlim(xbounds)
    ax[1][0].set_ylim(ybounds)
    ax[1][0].grid(grid)

    # X quadrature probability distribution
    ax[0][0].fill(q, ProbX, color=cm.RdBu(0.5))
    ax[0][0].plot(q, ProbX)
    ax[0][0].get_xaxis().set_ticks(ticks)
    ax[0][0].xaxis.set_ticklabels(tick_labels or [])
    ax[0][0].get_yaxis().set_ticks([])
    ax[0][0].tick_params(direction="in")
    ax[0][0].set_ylabel("Prob($x$)", fontsize=12)
    ax[0][0].set_xlim(xbounds)
    ax[0][0].set_ylim([0, 1.1 * max(ProbX)])
    ax[0][0].grid(grid)

    # P quadrature probability distribution
    ax[1][1].fill(ProbP, p, color=cm.RdBu(0.5))
    ax[1][1].plot(ProbP, p)
    ax[1][1].get_xaxis().set_ticks([])
    ax[1][1].get_yaxis().set_ticks(ticks)
    ax[1][1].yaxis.set_ticklabels(tick_labels or [])
    ax[1][1].tick_params(direction="in")
    ax[1][1].set_xlabel("Prob($p$)", fontsize=12)
    ax[1][1].set_xlim([0, 1.1 * max(ProbP)])
    ax[1][1].set_ylim(ybounds)
    ax[1][1].grid(grid)

    # Density matrix
    ax[0][1].matshow(abs(rho), cmap=cm.RdBu, vmin=-abs(rho).max(), vmax=abs(rho).max())
    ax[0][1].set_title(r"abs($\rho$)", fontsize=12)
    ax[0][1].tick_params(direction="in")
    ax[0][1].get_xaxis().set_ticks([])
    ax[0][1].get_yaxis().set_ticks([])
    ax[0][1].set_aspect("auto")
    ax[0][1].set_ylabel(f"cutoff = {len(rho)}", fontsize=12)
    ax[0][1].yaxis.set_label_position("right")
