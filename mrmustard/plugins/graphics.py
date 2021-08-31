from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import strawberryfields as sf  # TODO: remove dependency on strawberryfields
from mrmustard._typing import *
from mrmustard import Backend


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


def wigner(self, state, hbar: float = 2.0, filename: str = ""):
    r"""
    Plots the wigner function of a single mode state.
    Arguments:
        state (complex array): the state in Fock representation (can be pure or mixed)
        hbar (float): sets the scale of phase space (default 2.0)
        filename (str): optional filename for saving the plot of the wigner function
    """
    assert state.ndim in {1, 2}
    scale = np.sqrt(hbar)
    quad_axis = np.linspace(-6, 6, 200) * scale
    pure = state.ndim == 1  # if ndim=2, then it's density matrix
    state_sf = sf.backends.BaseFockState(state, 1, pure, len(state))  # TODO: remove dependency on strawberryfields
    Wig = state_sf.wigner(mode=0, xvec=quad_axis, pvec=quad_axis)
    scale = np.max(Wig.real)
    nrm = Normalize(-scale, scale)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    plt.contourf(quad_axis, quad_axis, Wig, 60, cmap=cm.RdBu, norm=nrm)
    plt.xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=15)
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename, dpi=300)
