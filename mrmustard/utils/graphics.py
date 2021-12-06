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

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from mrmustard.types import *
from mrmustard import settings


class Progressbar:
    "A spiffy loading bar to display the progress during an optimization"

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
        )

    def step(self, loss):
        speed = self.bar.tasks[0].speed or 0.0
        self.bar.update(self.taskID, advance=1, refresh=True, speed=speed, loss=loss)

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.bar.__exit__(exc_type, exc_val, exc_tb)


def wigner(state, filename: str = "", xbounds=(-6, 6), ybounds=(-6, 6)):
    r"""
    Plots the wigner function of a single mode state.
    Arguments:
        state (complex array): the state in Fock representation (can be pure or mixed)
        filename (str): optional filename for saving the plot of the wigner function
    """
    assert state.ndim in {1, 2}
    scale = np.sqrt(settings.HBAR)
    x_axis = np.linspace(*xbounds, 200) * scale
    y_axis = np.linspace(*ybounds, 200) * scale
    pure = state.ndim == 1  # if ndim=2, then it's density matrix
    state_sf = sf.backends.BaseFockState(
        state, 1, pure, len(state)
    )  # TODO: remove dependency on strawberryfields
    Wig = state_sf.wigner(mode=0, xvec=x_axis, pvec=y_axis)
    scale = np.max(Wig.real)
    nrm = Normalize(-scale, scale)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    plt.contourf(x_axis, y_axis, Wig, 60, cmap=cm.RdBu, norm=nrm)
    plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename, dpi=300)


def mikkel_plot(dm: np.ndarray, filename: str = "", xbounds=(-6, 6), ybounds=(-6, 6)):
    rho = dm.numpy()
    sf.hbar = settings.HBAR
    s = sf.ops.BaseFockState(rho, 1, False, rho.shape[0])
    X = np.linspace(xbounds[0], xbounds[1], 200)
    P = np.linspace(ybounds[0], ybounds[1], 200)
    W = s.wigner(0, X, P)
    ProbX = s.x_quad_values(0, X, P)
    ProbP = s.p_quad_values(0, X, P)

    ### PLOTTING ###

    fig, ax = plt.subplots(
        2, 2, figsize=(6, 6), gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]}
    )
    ticks = [-5, 0, 5]
    xlim = [X[0], X[-1]]
    plim = [P[0], P[-1]]
    grid = False
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Wigner function
    ax[1][0].contourf(X, P, W, 60, cmap=cm.RdBu, vmin=-abs(W).max(), vmax=abs(W).max())
    ax[1][0].set_xlabel("$x$", fontsize=12)
    ax[1][0].set_ylabel("$p$", fontsize=12)
    ax[1][0].get_xaxis().set_ticks(ticks)
    ax[1][0].get_yaxis().set_ticks(ticks)
    ax[1][0].tick_params(direction="in")
    ax[1][0].set_xlim(xlim)
    ax[1][0].set_ylim(plim)
    ax[1][0].grid(grid)

    # X quadrature probability distribution
    ax[0][0].fill(X, ProbX, color=cm.RdBu(0.5))
    ax[0][0].plot(X, ProbX)
    ax[0][0].get_xaxis().set_ticks(ticks)
    ax[0][0].xaxis.set_ticklabels([])
    ax[0][0].get_yaxis().set_ticks([])
    ax[0][0].tick_params(direction="in")
    ax[0][0].set_ylabel("Prob($x$)", fontsize=12)
    ax[0][0].set_xlim(xlim)
    ax[0][0].set_ylim([0, 1.1 * max(ProbX)])
    ax[0][0].grid(grid)

    # P quadrature probability distribution
    ax[1][1].fill(ProbP, P, color=cm.RdBu(0.5))
    ax[1][1].plot(ProbP, P)
    ax[1][1].get_xaxis().set_ticks([])
    ax[1][1].get_yaxis().set_ticks(ticks)
    ax[1][1].yaxis.set_ticklabels([])
    ax[1][1].tick_params(direction="in")
    ax[1][1].set_xlabel("Prob($p$)", fontsize=12)
    ax[1][1].set_xlim([0, 1.1 * max(ProbP)])
    ax[1][1].set_ylim(plim)
    ax[1][1].grid(grid)

    # Density matrix
    ax[0][1].matshow(abs(rho), cmap=cm.RdBu, vmin=-abs(rho).max(), vmax=abs(rho).max())
    ax[0][1].set_title(r"abs($\rho$)", fontsize=12)
    ax[0][1].tick_params(direction="in")
    ax[0][1].get_xaxis().set_ticks([])
    ax[0][1].get_yaxis().set_ticks([])
    ax[0][1].set_aspect("auto")  # if not set to auto, pytho
    ax[0][1].set_ylabel(f"cutoff = {len(rho)}", fontsize=12)
    ax[0][1].yaxis.set_label_position("right")
