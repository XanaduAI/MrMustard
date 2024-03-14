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

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mrmustard import math
from mrmustard.physics import fock
from mrmustard.physics.wigner import wigner_discretized
from mrmustard.utils.typing import ComplexMatrix

__all__ = ["mikkel_plot"]


def mikkel_plot(
    dm: ComplexMatrix,
    xbounds: tuple[int] = (-6, 6),
    ybounds: tuple[int] = (-6, 6),
    resolution: int = 200,
) -> go.Figure:
    r"""
    Visual representation of the Wigner function of a state given its density matrix.

    Args:
        dm: The density matrix to plot.
        xbounds: The range of the x axis.
        ybounds: The range of the y axis.
        resolution: The number of bins on each axes.

    Returns:
        The figure and axes.
    """
    x, prob_x = fock.quadrature_distribution(dm)
    p, prob_p = fock.quadrature_distribution(dm, np.pi / 2)

    xvec = np.linspace(*xbounds, resolution)
    pvec = np.linspace(*ybounds, resolution)
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
        # shared_xaxes=True
    )

    # X-P plot
    # note: heatmaps revert the y axes, which is why the minus in `y=-ps` is required
    fig_21 = go.Heatmap(x=xs, y=-ps, z=z, colorscale="viridis", name="Wigner function")
    fig.add_trace(fig_21, row=2, col=1)
    fig.update_traces(row=2, col=1, showscale=False)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="p", row=2, col=1)

    # X quadrature probability distribution
    fig_11 = go.Scatter(x=x, y=prob_x, line=dict(color="steelblue", width=2), name="Prob(x)")
    fig.add_trace(fig_11, row=1, col=1)
    fig.update_xaxes(range=xbounds, row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Prob(x)", range=(0, max(prob_x)), row=1, col=1)

    # P quadrature probability distribution
    fig_22 = go.Scatter(x=prob_p, y=-p, line=dict(color="steelblue", width=2), name="Prob(p)")
    fig.add_trace(fig_22, row=2, col=2)
    fig.update_xaxes(title_text="Prob(p)", range=(0, max(prob_p)), row=2, col=2)
    fig.update_yaxes(range=ybounds, row=2, col=2, showticklabels=False)

    # Density matrix
    fig_12 = go.Heatmap(z=abs(dm), colorscale="viridis", name=f"abs(ρ)")
    fig.add_trace(fig_12, row=1, col=2)
    fig.update_traces(showscale=False, row=1, col=2)
    fig.update_xaxes(row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        height=600,
        width=600,
        plot_bgcolor="aliceblue",
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, tickfont_family="Arial Black"
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, tickfont_family="Arial Black"
    )

    return fig
